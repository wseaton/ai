// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

use serde_json::{Value, json};

use super::*;
use crate::openai::responses::store::filter::extract_cancel_id;

// =============================================================================
// Helpers
// =============================================================================

fn request() -> Value {
    json!({"model": "m", "instructions": "Be brief.", "input": "Tell me about Paris.",
           "max_output_tokens": 64, "temperature": 0, "background": true,
           "tools": [{"type": "function", "name": "get_weather",
                      "parameters": {"type": "object", "properties": {}}}]})
}

fn queued_record() -> ResponseRecord {
    let request = request();
    let context = ResponseContext::from_responses_request(&request, "resp_1".to_owned(), 100);
    let mut response = in_progress_response_resource(&context).unwrap();
    set_status(&mut response, "queued");
    response["background"] = Value::Bool(true);
    ResponseRecord {
        id: "resp_1".to_owned(),
        owner: StateOwner::from_trusted_parts("tenant", "issuer", "subject").unwrap(),
        created_at: 100,
        model: "m".to_owned(),
        input: request["input"].clone(),
        messages: Value::Array(Vec::new()),
        response_object: response,
    }
}

fn result(status_code: u16, payload: &Value, error_code: Option<&str>) -> ProcessorResult {
    ProcessorResult {
        status_code,
        payload: if payload.is_null() {
            String::new()
        } else {
            payload.to_string()
        },
        error_code: error_code.map(str::to_owned),
        error_message: "detail".to_owned(),
    }
}

fn config(url: &str) -> BackgroundConfig {
    serde_json::from_value(json!({"processor_url": url})).unwrap()
}

// =============================================================================
// Request validation
// =============================================================================

#[test]
fn background_runs_stateless_requests_with_function_tools() {
    assert_eq!(unsupported(&request()), Ok(()));
    let mut explicit_nulls = request();
    explicit_nulls["previous_response_id"] = Value::Null;
    let mut continuation = request();
    continuation["previous_response_id"] = json!("resp_0");
    assert_eq!(unsupported(&continuation), Ok(()));
    explicit_nulls["conversation"] = Value::Null;
    explicit_nulls["store"] = Value::Bool(true);
    assert_eq!(unsupported(&explicit_nulls), Ok(()));
}

#[test]
fn background_refuses_what_it_does_not_run() {
    let cases = [
        (json!({"stream": true}), "background mode does not support stream=true"),
        (json!({"store": false}), "background mode requires store=true"),
        (
            json!({"conversation": "conv_0"}),
            "background mode does not support conversation",
        ),
        (
            json!({"tools": [{"type": "web_search"}]}),
            "background mode supports only function tools",
        ),
    ];
    for (extra, want) in cases {
        let mut body = request();
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        assert_eq!(unsupported(&body), Err(want), "{extra}");
    }
}

// =============================================================================
// Status and results
// =============================================================================

#[test]
fn status_of_stored_responses() {
    for (status, want) in [
        ("queued", Status::Queued),
        ("in_progress", Status::InProgress),
        ("completed", Status::Terminal),
        ("failed", Status::Terminal),
        ("cancelled", Status::Terminal),
        ("incomplete", Status::Terminal),
    ] {
        assert_eq!(Status::of(&json!({"status": status})), want, "{status}");
    }
    assert!(is_background(&queued_record().response_object));
    assert!(!is_background(&json!({"background": false})));
}

#[test]
fn a_chat_completion_result_completes_the_response() {
    let record = queued_record();
    let chat = json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Paris is the capital."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}});
    let response = terminal_response(&record, &result(200, &chat, None), 200, &ReasoningOptions::default()).unwrap();
    assert_eq!(response["id"], "resp_1");
    assert_eq!(response["status"], "completed");
    assert_eq!(response["background"], true);
    assert_eq!(response["created_at"], 100);
    assert_eq!(response["completed_at"], 200);
    assert_eq!(response["instructions"], "Be brief.");
    assert_eq!(response["tools"][0]["name"], "get_weather");
    assert_eq!(response["output"][0]["content"][0]["text"], "Paris is the capital.");
    assert_eq!(response["usage"]["output_tokens"], 5);
}

#[test]
fn a_truncated_result_is_incomplete() {
    let chat = json!({"id": "c", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "length",
                     "message": {"role": "assistant", "content": "Paris"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 64, "total_tokens": 74}});
    let response = terminal_response(
        &queued_record(),
        &result(200, &chat, None),
        200,
        &ReasoningOptions::default(),
    )
    .unwrap();
    assert_eq!(response["status"], "incomplete");
    assert_eq!(response["incomplete_details"]["reason"], "max_output_tokens");
}

#[test]
fn failures_map_to_openai_statuses() {
    let record = queued_record();
    let reasoning = ReasoningOptions::default();
    let cancelled = terminal_response(&record, &result(0, &Value::Null, Some("CANCELLED")), 200, &reasoning).unwrap();
    assert_eq!(cancelled["status"], "cancelled");
    assert_eq!(cancelled["error"], Value::Null);

    for (code, want) in [
        (Some("DEADLINE_EXCEEDED"), "timeout"),
        (Some("GATE_DROPPED"), "rate_limit_exceeded"),
        (Some("INVALID_REQUEST"), "invalid_prompt"),
        (Some("INFERENCE_ERROR"), "server_error"),
        (None, "server_error"),
    ] {
        let failed = terminal_response(&record, &result(0, &Value::Null, code), 200, &reasoning).unwrap();
        assert_eq!(failed["status"], "failed", "{code:?}");
        assert_eq!(failed["error"], json!({"code": want, "message": "detail"}), "{code:?}");
    }

    let upstream = |status| {
        result(
            status,
            &json!({"error": {"message": "bad input", "code": status}}),
            None,
        )
    };
    let client = terminal_response(&record, &upstream(400), 200, &reasoning).unwrap();
    assert_eq!(
        client["error"],
        json!({"code": "invalid_prompt", "message": "bad input"})
    );
    let server = terminal_response(&record, &upstream(500), 200, &reasoning).unwrap();
    assert_eq!(server["error"], json!({"code": "server_error", "message": "bad input"}));

    assert!(terminal_response(&record, &result(200, &json!("not a completion"), None), 200, &reasoning).is_err());
    let unparsable = ProcessorResult {
        payload: "{".to_owned(),
        ..result(200, &Value::Null, None)
    };
    assert!(terminal_response(&record, &unparsable, 200, &reasoning).is_err());
}

// =============================================================================
// Configuration and paths
// =============================================================================

#[test]
fn config_defaults_and_validation() {
    let cfg = config("https://processor.example.com");
    assert_eq!(cfg.deadline_secs, 3_600);
    assert_eq!(cfg.timeout_ms, 10_000);
    assert!(validate_background(&cfg).is_ok());

    assert!(validate_background(&config("http://127.0.0.1:8080")).is_err());
    let mut private = config("http://127.0.0.1:8080");
    private.allow_private_processor_url = true;
    assert!(validate_background(&private).is_ok());

    for field in ["deadline_secs", "timeout_ms"] {
        let cfg: BackgroundConfig =
            serde_json::from_value(json!({"processor_url": "https://p.example.com", field: 0})).unwrap();
        let err = validate_background(&cfg).unwrap_err().to_string();
        assert!(err.contains(field), "{err}");
    }
    assert!(serde_json::from_value::<BackgroundConfig>(json!({"processor_url": "x", "nope": 1})).is_err());
    assert!(validate_background(&config("ftp://p.example.com")).is_err());
}

#[test]
fn cancel_paths() {
    assert_eq!(extract_cancel_id("/v1/responses/resp_1/cancel"), Some("resp_1"));
    assert_eq!(extract_cancel_id("/v1/responses/resp_1/cancel/"), Some("resp_1"));
    for path in [
        "/v1/responses/cancel",
        "/v1/responses//cancel",
        "/v1/responses/a/b/cancel",
        "/v1/responses/resp_1",
        "/v1/responses/resp_1/input_items",
    ] {
        assert_eq!(extract_cancel_id(path), None, "{path}");
    }
}

// =============================================================================
// Compatibility options
// =============================================================================

fn prepared(config: &serde_json::Value, extra: &serde_json::Value) -> Result<Queued, FilterAction> {
    let mut cfg = json!({"processor_url": "http://127.0.0.1:1", "allow_private_processor_url": true});
    cfg.as_object_mut().unwrap().extend(config.as_object().unwrap().clone());
    let background = Background::new(serde_json::from_value(cfg).unwrap());
    let mut body = request();
    body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    let http_request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let ctx = crate::test_utils::make_filter_context(&http_request);
    background.prepare(&ctx, &body, None)
}

fn rejected(result: Result<Queued, FilterAction>) -> u16 {
    match result {
        Err(FilterAction::Reject(rejection)) => rejection.status,
        Err(other) => panic!("expected a rejection, got {other:?}"),
        Ok(_) => panic!("expected a rejection"),
    }
}

#[test]
fn reasoning_summaries_follow_the_dialect_options() {
    let summary = json!({"reasoning": {"summary": "auto"}});
    assert_eq!(
        rejected(prepared(&json!({"reasoning": {"dialect": "vllm"}}), &summary)),
        400
    );
    let queued = prepared(&json!({"reasoning": {"dialect": "vllm", "summary": "omit"}}), &summary)
        .unwrap_or_else(|_| panic!("an omitted summary runs"));
    assert_eq!(queued.response["status"], "queued");
    assert!(queued.chat.get("reasoning").is_none() || queued.chat["reasoning"].get("summary").is_none());
}

#[test]
fn auto_truncation_follows_the_option() {
    let auto = json!({"truncation": "auto"});
    assert_eq!(rejected(prepared(&json!({}), &auto)), 400);
    let queued = prepared(&json!({"truncation_auto": "disabled"}), &auto)
        .unwrap_or_else(|_| panic!("auto truncation runs as disabled"));
    assert_eq!(queued.response["truncation"], "disabled");
}

#[test]
fn a_continuation_translates_the_stored_history_before_the_new_input() {
    let background = Background::new(config("https://p.example.com"));
    let mut previous = queued_record();
    set_status(&mut previous.response_object, "completed");
    previous.messages = json!([
        {"type": "message", "role": "user", "content": "Tell me about Paris."},
        {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}"}
    ]);
    let mut body = request();
    body["previous_response_id"] = json!("resp_1");
    body["input"] = json!([{"type": "function_call_output", "call_id": "call_1", "output": "17C"}]);
    let http_request = crate::test_utils::make_request(http::Method::POST, "/v1/responses");
    let ctx = crate::test_utils::make_filter_context(&http_request);

    let queued = background
        .prepare(&ctx, &body, Some(previous.clone()))
        .unwrap_or_else(|_| panic!("a continuation of a completed response runs"));
    let roles: Vec<&str> = queued.chat["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    assert_eq!(queued.chat["messages"][2]["tool_calls"][0]["id"], "call_1");
    assert_eq!(queued.chat["messages"][3]["content"], "17C");
    assert_eq!(queued.response["previous_response_id"], "resp_1");
    assert_eq!(queued.history.as_ref().map(Vec::len), Some(3));

    set_status(&mut previous.response_object, "in_progress");
    assert_eq!(rejected(background.prepare(&ctx, &body, Some(previous))), 400);
}
