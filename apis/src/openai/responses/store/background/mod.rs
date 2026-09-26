// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Background Responses (`background: true`) run by an asynchronous
//! inference processor ([llm-d-async]).
//!
//! ```text
//!  POST /v1/responses {background: true}
//!    ├─ translate to Chat Completions
//!    ├─ submit to the processor (result delivered by request id) ──► queue ──► inference
//!    └─ store and return the `queued` response
//!
//!  GET /v1/responses/{id}        (queued or in_progress)
//!    ├─ claim the result from the processor
//!    │    └─ found: translate, store the terminal response, acknowledge
//!    └─ otherwise: store the processor's queued / in_progress status
//!
//!  POST /v1/responses/{id}/cancel
//!    └─ cancel at the processor, store and return the `cancelled` response
//! ```
//!
//! The processor keeps the request durable and deadline-ordered, retries it,
//! and resumes interrupted generations, so a background response survives
//! Praxis restarts: the stored record and the processor's result route are
//! all the state there is.
//!
//! [llm-d-async]: https://github.com/wseaton/llm-d-async-rs

#[cfg(test)]
#[expect(clippy::allow_attributes, reason = "blanket test suppressions")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::too_many_lines,
    reason = "tests"
)]
mod tests;

use std::time::Duration;

use bytes::Bytes;
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use praxis_filter::{FilterAction, FilterError, HttpFilterContext, Rejection};
use serde::Deserialize;
use serde_json::{Value, json};
use tracing::{debug, warn};

use super::filter::assemble_stored_messages;
use crate::{
    callout_target::{AddressPolicy, validate_configured_http_target},
    openai::{
        responses::{error::responses_error_rejection, rehydrate::continuation_state},
        translation::{
            chat_completions::{
                ResponseContext, TruncationAuto, chat_response_to_response_resource, in_progress_response_resource,
                responses_request_to_chat_request, responses_state_to_chat_request,
            },
            reasoning::{ReasoningOptions, validate_requested_reasoning},
        },
    },
    state_owner::StateOwner,
    store::{ResponseRecord, ResponseStore},
    subrequest::{self, SubRequest, SubRequestClient},
};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------

/// Filter name used in configuration errors.
const FILTER_NAME: &str = "openai_response_store";

/// Default time a background response may take before it fails.
const DEFAULT_DEADLINE_SECS: u64 = 3_600;

/// Default timeout of one processor call.
const DEFAULT_TIMEOUT_MS: u64 = 10_000;

/// How long a claimed result stays leased while it is translated and stored.
const RESULT_LEASE_MS: u64 = 60_000;

/// Largest processor reply read.
const MAX_PROCESSOR_RESPONSE_BYTES: usize = 64 * 1024 * 1024;

/// Header carrying the inference objective to the inference gateway.
const OBJECTIVE_HEADER: &str = "x-llm-d-inference-objective";

/// Characters escaped in a URL path segment: all but RFC 3986 unreserved.
const SEGMENT: &AsciiSet = &NON_ALPHANUMERIC.remove(b'-').remove(b'_').remove(b'.').remove(b'~');

/// Path the processor sends a background request to.
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// `background` section of the `openai_response_store` configuration.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BackgroundConfig {
    /// Base URL of the processor API, for example `http://llm-d-async:8080`.
    pub processor_url: String,

    /// Processor queue for background requests. Defaults to the
    /// processor's first queue.
    #[serde(default)]
    pub queue: Option<String>,

    /// Inference objective for background requests, sent to the inference
    /// gateway as `x-llm-d-inference-objective`.
    #[serde(default)]
    pub objective: Option<String>,

    /// Seconds a background response may take before it fails.
    #[serde(default = "default_deadline_secs")]
    pub deadline_secs: u64,

    /// Timeout of one processor call, in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// Allow a processor URL on a private, loopback, or link-local address.
    #[serde(default)]
    pub allow_private_processor_url: bool,

    /// Reasoning dialect of the Chat Completions backend behind the
    /// processor.
    #[serde(default)]
    pub reasoning: ReasoningOptions,

    /// Handling of `truncation: "auto"`.
    #[serde(default)]
    pub truncation_auto: TruncationAuto,
}

/// Default for [`BackgroundConfig::deadline_secs`].
const fn default_deadline_secs() -> u64 {
    DEFAULT_DEADLINE_SECS
}

/// Default for [`BackgroundConfig::timeout_ms`].
const fn default_timeout_ms() -> u64 {
    DEFAULT_TIMEOUT_MS
}

/// Validate the `background` section.
pub(crate) fn validate_background(config: &BackgroundConfig) -> Result<(), FilterError> {
    let policy = AddressPolicy::from_allow_private(config.allow_private_processor_url);
    validate_configured_http_target(FILTER_NAME, &config.processor_url, policy)?;
    if config.deadline_secs == 0 {
        return Err(format!("{FILTER_NAME}: background.deadline_secs must be greater than 0").into());
    }
    if config.timeout_ms == 0 {
        return Err(format!("{FILTER_NAME}: background.timeout_ms must be greater than 0").into());
    }
    Ok(())
}

// -----------------------------------------------------------------------------
// Background
// -----------------------------------------------------------------------------

/// Runs background responses through the processor.
pub(crate) struct Background {
    /// Processor API base URL, without a trailing slash.
    processor_url: String,
    /// Validated configuration.
    config: BackgroundConfig,
    /// Address policy for processor calls.
    address_policy: AddressPolicy,
    /// Client for processor calls.
    client: SubRequestClient,
}

/// Where a background response stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Waiting in the processor's queue.
    Queued,
    /// Being generated.
    InProgress,
    /// Finished, failed, incomplete, or cancelled.
    Terminal,
}

impl Status {
    /// The status of a stored response object.
    fn of(response: &Value) -> Self {
        match response.get("status").and_then(Value::as_str) {
            Some("queued") => Self::Queued,
            Some("in_progress") => Self::InProgress,
            _ => Self::Terminal,
        }
    }
}

/// A processor result, as `llm-d-async` delivers it.
#[derive(Debug, Deserialize)]
struct ProcessorResult {
    /// HTTP status of the inference response, 0 when there was none.
    #[serde(default)]
    status_code: u16,
    /// The inference response body.
    #[serde(default)]
    payload: String,
    /// Why there was no inference response.
    #[serde(default)]
    error_code: Option<String>,
    /// Detail for `error_code`.
    #[serde(default)]
    error_message: String,
}

/// A leased processor result.
#[derive(Debug, Deserialize)]
struct Claim {
    /// Lease identifier, used to acknowledge the result.
    claim_id: u64,
    /// Lease owner, used to acknowledge the result.
    owner_token: String,
    /// The result.
    result: ProcessorResult,
}

/// The processor's view of a request.
#[derive(Debug, Deserialize)]
struct RequestState {
    /// `queued`, `in_progress`, or `done`.
    status: String,
}

/// A background response ready to submit.
struct Queued {
    /// Response ID, also the processor request ID.
    id: String,
    /// Creation time, Unix seconds.
    created_at: u64,
    /// Model named by the request.
    model: String,
    /// The Chat Completions request the processor runs.
    chat: Value,
    /// The `queued` response resource.
    response: Value,
    /// The previous response's history followed by this input, for a
    /// continuation.
    history: Option<Vec<Value>>,
}

impl Background {
    /// Build from a validated configuration.
    pub(crate) fn new(config: BackgroundConfig) -> Self {
        Self {
            processor_url: config.processor_url.trim_end_matches('/').to_owned(),
            address_policy: AddressPolicy::from_allow_private(config.allow_private_processor_url),
            config,
            client: subrequest::isolated_client(8),
        }
    }

    /// Call the processor.
    async fn call(&self, method: http::Method, path: &str, body: Option<&Value>) -> Result<(u16, Bytes), String> {
        let request = processor_request(method, body)?;
        let url = format!("{}{path}", self.processor_url);
        Box::pin(subrequest::execute_url(
            &self.client,
            &url,
            request,
            MAX_PROCESSOR_RESPONSE_BYTES,
            Duration::from_millis(self.config.timeout_ms),
            self.address_policy,
        ))
        .await
        .map(|response| (response.status, response.body))
        .map_err(|e| e.to_string())
    }

    /// Create a background response for the Responses create `request`.
    pub(crate) async fn create(
        &self,
        ctx: &HttpFilterContext<'_>,
        store: &dyn ResponseStore,
        owner: &StateOwner,
        request: &Value,
    ) -> FilterAction {
        let previous = match Box::pin(previous_response(store, owner, request)).await {
            Ok(previous) => previous,
            Err(action) => return action,
        };
        let queued = match self.prepare(ctx, request, previous) {
            Ok(queued) => queued,
            Err(action) => return action,
        };
        if let Err(action) = Box::pin(self.submit(&queued)).await {
            return action;
        }
        Box::pin(self.store_queued(store, owner, request, queued)).await
    }

    /// Translate `request`, continuing the stored `previous` response when
    /// it names one, and build its `queued` response.
    fn prepare(
        &self,
        ctx: &HttpFilterContext<'_>,
        request: &Value,
        previous: Option<ResponseRecord>,
    ) -> Result<Queued, FilterAction> {
        unsupported(request).map_err(|message| reject(400, "invalid_request_error", message))?;
        let mut foreground = request.clone();
        if let Some(fields) = foreground.as_object_mut() {
            fields.remove("background");
            validate_requested_reasoning(fields, &self.config.reasoning)
                .map_err(|e| reject(400, "invalid_request_error", &e.to_string()))?;
        }
        self.config.truncation_auto.apply(&mut foreground);
        let (chat, history) = self.translate(foreground, previous)?;
        let id = format!("resp_{}", ctx.id_generator.generate(ctx.time_source));
        let created_at = ctx.time_source.now().as_secs();
        let context = ResponseContext::from_responses_request(request, id.clone(), created_at);
        let mut response = in_progress_response_resource(&context)
            .map_err(|e| reject(400, "invalid_request_error", &e.to_string()))?;
        set_status(&mut response, "queued");
        set_field(&mut response, "background", Value::Bool(true));
        Ok(Queued {
            model: context.model.to_owned(),
            id,
            created_at,
            chat,
            response,
            history,
        })
    }

    /// The Chat Completions request for `request`, after the stored history
    /// of `previous` when it continues one, and that history followed by the
    /// new input.
    fn translate(
        &self,
        request: Value,
        previous: Option<ResponseRecord>,
    ) -> Result<(Value, Option<Vec<Value>>), FilterAction> {
        let translated = match previous {
            Some(previous) => {
                let state = continuation_state(request, previous)?;
                let chat = responses_state_to_chat_request(
                    &state.request_body,
                    &state.messages,
                    state.request_tools(),
                    state.request_tool_choice(),
                    &self.config.reasoning,
                );
                chat.map(|chat| (chat, Some(state.persisted_messages)))
            },
            None => responses_request_to_chat_request(&request, &self.config.reasoning).map(|chat| (chat, None)),
        };
        translated.map_err(|e| reject(400, "invalid_request_error", &e.to_string()))
    }

    /// The processor submission of `queued`, its result delivered by
    /// request ID.
    fn submission(&self, queued: &Queued) -> Value {
        let mut headers = serde_json::Map::new();
        if let Some(objective) = &self.config.objective {
            headers.insert(OBJECTIVE_HEADER.to_owned(), Value::String(objective.clone()));
        }
        json!({
            "id": queued.id,
            "created": queued.created_at,
            "deadline": queued.created_at.saturating_add(self.config.deadline_secs),
            "endpoint": CHAT_COMPLETIONS_PATH,
            "model": queued.model,
            "headers": headers,
            "request_queue_name": self.config.queue.as_deref().unwrap_or_default(),
            "result_delivery": "request",
            "payload": queued.chat,
        })
    }

    /// Submit `queued` to the processor.
    async fn submit(&self, queued: &Queued) -> Result<(), FilterAction> {
        let submission = self.submission(queued);
        match Box::pin(self.call(http::Method::POST, "/v1/requests", Some(&submission))).await {
            Ok((202, _)) => Ok(()),
            Ok((status, body)) => {
                warn!(status, body = %String::from_utf8_lossy(&body), "processor refused a background request");
                Err(reject(
                    502,
                    "server_error",
                    "the background processor refused the request",
                ))
            },
            Err(e) => {
                warn!(error = %e, "background submission failed");
                Err(reject(503, "server_error", "the background processor is unavailable"))
            },
        }
    }

    /// Store the submitted `queued` response and return it.
    #[expect(clippy::cognitive_complexity, reason = "tracing macros inflate complexity")]
    async fn store_queued(
        &self,
        store: &dyn ResponseStore,
        owner: &StateOwner,
        request: &Value,
        queued: Queued,
    ) -> FilterAction {
        let input = request.get("input").cloned().unwrap_or(Value::Null);
        let history = queued.history.map_or_else(|| input.clone(), Value::Array);
        let record = ResponseRecord {
            id: queued.id,
            owner: owner.clone(),
            created_at: i64::try_from(queued.created_at).unwrap_or(i64::MAX),
            model: queued.model,
            messages: assemble_stored_messages(history, None),
            input,
            response_object: queued.response,
        };
        if let Err(e) = store.upsert_response(&record).await {
            warn!(error = %e, "failed to store a background response");
            if let Err(e) = Box::pin(self.cancel_at_processor(&record.id)).await {
                warn!(error = %e, "failed to cancel an unstored background response");
            }
            return reject(500, "server_error", "failed to store the background response");
        }
        debug!(response_id = %record.id, "background response queued");
        respond(&record.response_object)
    }

    /// Bring a queued or in-progress background `record` up to date with the
    /// processor at `now` (Unix seconds). A processor that cannot be reached
    /// leaves it as stored.
    pub(crate) async fn refresh(
        &self,
        store: &dyn ResponseStore,
        mut record: ResponseRecord,
        now: u64,
    ) -> ResponseRecord {
        if !is_background(&record.response_object) || Status::of(&record.response_object) == Status::Terminal {
            return record;
        }
        match Box::pin(self.claim(&record.id)).await {
            Ok(Some(claim)) => Box::pin(self.complete(store, &mut record, claim, now)).await,
            Ok(None) => {
                if Box::pin(self.track(&mut record)).await
                    && let Err(e) = store.upsert_response(&record).await
                {
                    warn!(response_id = %record.id, error = %e, "failed to store a background status");
                }
            },
            Err(e) => warn!(response_id = %record.id, error = %e, "failed to reach the background processor"),
        }
        record
    }

    /// Complete `record` from a claimed result. The result is acknowledged
    /// only once the response is stored; otherwise its lease lapses and a
    /// later poll completes it again.
    async fn complete(&self, store: &dyn ResponseStore, record: &mut ResponseRecord, claim: Claim, now: u64) {
        let response = match terminal_response(record, &claim.result, now, &self.config.reasoning) {
            Ok(response) => response,
            Err(e) => {
                warn!(response_id = %record.id, error = %e, "untranslatable background result");
                failed_response(&record.response_object, "server_error", &e)
            },
        };
        record.messages = assemble_stored_messages(std::mem::take(&mut record.messages), response.get("output"));
        record.response_object = response;
        match store.upsert_response(record).await {
            Ok(()) => Box::pin(self.acknowledge(&record.id, claim.claim_id, &claim.owner_token)).await,
            Err(e) => warn!(response_id = %record.id, error = %e, "failed to store a finished background response"),
        }
    }

    /// Cancel a background `record`: queued or in progress, it stops at the
    /// processor and becomes `cancelled`; terminal, it is returned as is.
    pub(crate) async fn cancel(&self, store: &dyn ResponseStore, record: ResponseRecord, now: u64) -> FilterAction {
        if !is_background(&record.response_object) {
            return reject(
                400,
                "invalid_request_error",
                "only responses created with background=true can be cancelled",
            );
        }
        let mut record = Box::pin(self.refresh(store, record, now)).await;
        if Status::of(&record.response_object) == Status::Terminal {
            return respond(&record.response_object);
        }
        if let Err(e) = Box::pin(self.cancel_at_processor(&record.id)).await {
            warn!(response_id = %record.id, error = %e, "failed to cancel at the background processor");
        }
        set_status(&mut record.response_object, "cancelled");
        if let Err(e) = store.upsert_response(&record).await {
            warn!(response_id = %record.id, error = %e, "failed to store a cancelled background response");
            return reject(500, "server_error", "failed to store the cancelled response");
        }
        respond(&record.response_object)
    }

    /// Lease the result of request `id`, if there is one.
    async fn claim(&self, id: &str) -> Result<Option<Claim>, String> {
        let path = format!("{}/claims?wait_ms=0&lease_ms={RESULT_LEASE_MS}", result_route(id));
        match Box::pin(self.call(http::Method::POST, &path, None)).await? {
            (204, _) => Ok(None),
            (200, body) => serde_json::from_slice(&body).map(Some).map_err(|e| e.to_string()),
            (status, _) => Err(format!("claiming a result returned {status}")),
        }
    }

    /// Acknowledge a stored result so the processor deletes it.
    async fn acknowledge(&self, id: &str, claim_id: u64, owner_token: &str) {
        let path = format!("{}/claims/{claim_id}/ack", result_route(id));
        let body = json!({"owner_token": owner_token});
        match Box::pin(self.call(http::Method::POST, &path, Some(&body))).await {
            Ok((204 | 200, _)) => {},
            Ok((status, _)) => warn!(response_id = id, status, "acknowledging a background result failed"),
            Err(e) => warn!(response_id = id, error = %e, "acknowledging a background result failed"),
        }
    }

    /// Cancel request `id` at the processor, before it is dispatched.
    async fn cancel_at_processor(&self, id: &str) -> Result<(), String> {
        let body = json!({"ids": [id]});
        match Box::pin(self.call(http::Method::POST, "/v1/requests/cancel", Some(&body))).await? {
            (200, _) => Ok(()),
            (status, _) => Err(format!("cancel returned {status}")),
        }
    }

    /// Move a still-running `record` to the processor's status. Returns
    /// whether it changed.
    async fn track(&self, record: &mut ResponseRecord) -> bool {
        let path = format!("/v1/requests/{}", utf8_percent_encode(&record.id, SEGMENT));
        let status = match Box::pin(self.call(http::Method::GET, &path, None)).await {
            Ok((200, body)) => serde_json::from_slice::<RequestState>(&body)
                .map(|s| s.status)
                .map_err(|e| e.to_string()),
            Ok((status, _)) => Err(format!("status lookup returned {status}")),
            Err(e) => Err(e),
        };
        match status {
            Ok(status) if status == "in_progress" && Status::of(&record.response_object) == Status::Queued => {
                set_status(&mut record.response_object, "in_progress");
                true
            },
            Ok(_) => false,
            Err(e) => {
                warn!(response_id = %record.id, error = %e, "background status lookup failed");
                false
            },
        }
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// A processor request with an optional JSON body.
fn processor_request(method: http::Method, body: Option<&Value>) -> Result<SubRequest, String> {
    let mut headers = http::HeaderMap::new();
    headers.insert(http::header::ACCEPT, http::HeaderValue::from_static("application/json"));
    let body = match body {
        Some(body) => {
            headers.insert(
                http::header::CONTENT_TYPE,
                http::HeaderValue::from_static("application/json"),
            );
            Bytes::from(serde_json::to_vec(body).map_err(|e| e.to_string())?)
        },
        None => Bytes::new(),
    };
    Ok(SubRequest {
        method,
        uri: http::Uri::default(),
        headers,
        body,
    })
}

/// The processor path of request `id`'s result route.
fn result_route(id: &str) -> String {
    format!(
        "/v1/results/{}",
        utf8_percent_encode(&format!("@request-{id}"), SEGMENT)
    )
}

/// The stored response a request's `previous_response_id` names, if any.
async fn previous_response(
    store: &dyn ResponseStore,
    owner: &StateOwner,
    request: &Value,
) -> Result<Option<ResponseRecord>, FilterAction> {
    let Some(id) = request.get("previous_response_id").filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let Some(id) = id.as_str() else {
        return Err(reject(
            400,
            "invalid_request_error",
            "previous_response_id must be a string",
        ));
    };
    match store.get_response(owner, id).await {
        Ok(Some(record)) => Ok(Some(record)),
        Ok(None) => Err(reject(
            400,
            "invalid_request_error",
            &format!("previous response '{id}' not found"),
        )),
        Err(e) => {
            warn!(error = %e, "previous response lookup failed");
            Err(reject(500, "server_error", "failed to load the previous response"))
        },
    }
}

/// Whether a stored response object is a background response.
pub(crate) fn is_background(response: &Value) -> bool {
    response.get("background").and_then(Value::as_bool) == Some(true)
}

/// Reject what background mode does not run: streams, server-held state,
/// and server-side tools, which need the proxy's agentic loop.
fn unsupported(request: &Value) -> Result<(), &'static str> {
    if request.get("stream").and_then(Value::as_bool) == Some(true) {
        return Err("background mode does not support stream=true");
    }
    if request.get("store").and_then(Value::as_bool) == Some(false) {
        return Err("background mode requires store=true");
    }
    if request.get("conversation").is_some_and(|v| !v.is_null()) {
        return Err("background mode does not support conversation");
    }
    let server_tools = request.get("tools").and_then(Value::as_array).is_some_and(|tools| {
        tools
            .iter()
            .any(|tool| tool.get("type").and_then(Value::as_str) != Some("function"))
    });
    if server_tools {
        return Err("background mode supports only function tools");
    }
    Ok(())
}

/// The response a terminal processor `result` gives.
fn terminal_response(
    record: &ResponseRecord,
    result: &ProcessorResult,
    now: u64,
    reasoning: &ReasoningOptions,
) -> Result<Value, String> {
    let queued = &record.response_object;
    if result.status_code == 0 {
        return Ok(unanswered(queued, result));
    }
    let body: Value = serde_json::from_str(&result.payload).map_err(|e| e.to_string())?;
    if !(200..300).contains(&result.status_code) {
        return Ok(upstream_failure(queued, result.status_code, &body));
    }
    let created_at = queued.get("created_at").and_then(Value::as_u64).unwrap_or(now);
    let mut context = ResponseContext::from_responses_request(queued, record.id.clone(), created_at);
    context.completed_at = Some(now);
    context.reasoning_options = reasoning.clone();
    let mut response = chat_response_to_response_resource(&body, &context).map_err(|e| e.to_string())?;
    set_field(&mut response, "background", Value::Bool(true));
    Ok(response)
}

/// The response of a request the processor finished without an inference
/// response: cancelled, or failed.
fn unanswered(queued: &Value, result: &ProcessorResult) -> Value {
    if result.error_code.as_deref() == Some("CANCELLED") {
        let mut response = queued.clone();
        set_status(&mut response, "cancelled");
        return response;
    }
    failed_response(queued, error_code(result.error_code.as_deref()), &result.error_message)
}

/// The response of an inference that answered with an error `status`.
fn upstream_failure(queued: &Value, status: u16, body: &Value) -> Value {
    let message = body
        .pointer("/error/message")
        .and_then(Value::as_str)
        .unwrap_or("the background inference failed");
    let code = if (400..500).contains(&status) {
        "invalid_prompt"
    } else {
        "server_error"
    };
    failed_response(queued, code, message)
}

/// The OpenAI error code for a processor error code.
fn error_code(code: Option<&str>) -> &'static str {
    match code {
        Some("DEADLINE_EXCEEDED") => "timeout",
        Some("GATE_DROPPED") => "rate_limit_exceeded",
        Some("INVALID_REQUEST") => "invalid_prompt",
        _ => "server_error",
    }
}

/// `queued` as a failed response.
fn failed_response(queued: &Value, code: &str, message: &str) -> Value {
    let mut response = queued.clone();
    set_status(&mut response, "failed");
    set_field(&mut response, "error", json!({"code": code, "message": message}));
    response
}

/// Set a response object's `status`.
fn set_status(response: &mut Value, status: &str) {
    set_field(response, "status", Value::String(status.to_owned()));
}

/// Set a field of a response object.
fn set_field(response: &mut Value, name: &str, value: Value) {
    if let Some(fields) = response.as_object_mut() {
        fields.insert(name.to_owned(), value);
    }
}

/// A 200 JSON response carrying `response`.
fn respond(response: &Value) -> FilterAction {
    let body = serde_json::to_vec(response).unwrap_or_default();
    FilterAction::Reject(
        Rejection::status(200)
            .with_header("content-type", "application/json")
            .with_body(body),
    )
}

/// An OpenAI-shaped error response.
fn reject(status: u16, code: &str, message: &str) -> FilterAction {
    FilterAction::Reject(responses_error_rejection(status, code, message))
}
