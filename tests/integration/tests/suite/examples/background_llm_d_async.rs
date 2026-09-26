// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Praxis Contributors

//! Functional tests for the llm-d-async background-mode example config.
//!
//! The backend stands in for the llm-d-async processor: it answers the
//! processor calls Praxis makes, in order, and records them.

use std::collections::HashMap;

use praxis_test_utils::{
    CapturedRequest, StatefulCapturingBackend, TempSqlite, example_config_path, free_port, http_get, http_send,
    json_post, parse_body, parse_status, patch_yaml, start_proxy,
};
use serde_json::{Value, json};

const EXAMPLE: &str = "openai/responses/background-llm-d-async.yaml";

fn load_test_config(
    test_name: &str,
    listener_port: u16,
    processor_port: u16,
) -> (praxis_core::config::Config, TempSqlite) {
    let db = TempSqlite::new(test_name);
    let yaml = std::fs::read_to_string(example_config_path(EXAMPLE)).expect("example config should exist");
    let patched = patch_yaml(
        &yaml.replace("sqlite://responses.db?mode=rwc", db.url()),
        listener_port,
        &HashMap::from([("127.0.0.1:3001", processor_port)]),
    );
    let config = praxis_core::config::Config::from_yaml(&patched).expect("patched config should parse");
    (config, db)
}

fn create_request() -> Value {
    json!({
        "model": "m",
        "instructions": "Be concise.",
        "input": "Tell me about Paris.",
        "background": true,
        "tools": [{"type": "function", "name": "get_weather",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    })
}

fn submitted(id: &str) -> (u16, String) {
    (
        202,
        json!({"id": id, "request_token": "t", "result_route": format!("@request-{id}")}).to_string(),
    )
}

fn json_of(raw: &str) -> Value {
    serde_json::from_str(&parse_body(raw)).expect("response should be JSON")
}

fn retrieve(addr: &str, id: &str) -> Value {
    let (status, body) = http_get(addr, &format!("/v1/responses/{id}"), None);
    assert_eq!(status, 200, "{body}");
    serde_json::from_str(&body).expect("response should be JSON")
}

fn body_of(request: &CapturedRequest) -> Value {
    serde_json::from_str(&request.body).expect("processor request should be JSON")
}

fn route_of(id: &str) -> String {
    format!("/v1/results/%40request-{id}")
}

#[test]
fn background_response_runs_through_the_processor() {
    let completion = json!({
        "id": "chatcmpl_1", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Paris is the capital of France."}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}
    });
    let claim = json!({"claim_id": 7, "owner_token": "owner", "lease_ms": 60000,
                       "result": {"id": "x", "status_code": 200, "payload": completion.to_string()}});
    let processor = StatefulCapturingBackend::new(vec![
        submitted("placeholder"),
        (204, String::new()),
        (200, json!({"id": "x", "status": "in_progress"}).to_string()),
        (200, claim.to_string()),
        (204, String::new()),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config("background_lifecycle", proxy_port, processor.port());
    let proxy = start_proxy(&config);

    let created = http_send(proxy.addr(), &json_post("/v1/responses", &create_request().to_string()));
    assert_eq!(parse_status(&created), 200, "{created}");
    let created = json_of(&created);
    let id = created["id"].as_str().expect("response id").to_owned();
    assert!(id.starts_with("resp_"), "{id}");
    assert_eq!(created["status"], "queued");
    assert_eq!(created["background"], true);
    assert_eq!(created["output"], json!([]));

    let polled = retrieve(proxy.addr(), &id);
    assert_eq!(polled["status"], "in_progress");

    let completed = retrieve(proxy.addr(), &id);
    assert_eq!(completed["status"], "completed", "{completed}");
    assert_eq!(completed["id"], id.as_str());
    assert_eq!(completed["background"], true);
    assert_eq!(completed["instructions"], "Be concise.");
    assert_eq!(
        completed["output"][0]["content"][0]["text"],
        "Paris is the capital of France."
    );
    assert_eq!(completed["usage"]["output_tokens"], 7);

    let again = retrieve(proxy.addr(), &id);
    assert_eq!(again, completed, "a terminal response is served from the store");

    let calls = processor.requests();
    assert_eq!(calls.len(), 5, "{:?}", calls.iter().map(|c| &c.uri).collect::<Vec<_>>());

    assert_eq!(
        (calls[0].method.as_str(), calls[0].uri.as_str()),
        ("POST", "/v1/requests")
    );
    let submission = body_of(&calls[0]);
    assert_eq!(submission["id"], id.as_str());
    assert_eq!(submission["endpoint"], "/v1/chat/completions");
    assert_eq!(submission["result_delivery"], "request");
    assert_eq!(submission["headers"], json!({"x-llm-d-inference-objective": "batch"}));
    assert_eq!(submission["model"], "m");
    assert!(submission["deadline"].as_u64() > submission["created"].as_u64());
    assert_eq!(
        submission["payload"]["messages"][0],
        json!({"role": "system", "content": "Be concise."})
    );
    assert_eq!(submission["payload"]["tools"][0]["function"]["name"], "get_weather");
    assert!(submission["payload"].get("background").is_none());

    let claims = format!("{}/claims?wait_ms=0&lease_ms=60000", route_of(&id));
    assert_eq!(
        (calls[1].method.as_str(), calls[1].uri.as_str()),
        ("POST", claims.as_str())
    );
    assert_eq!(
        (calls[2].method.as_str(), calls[2].uri.as_str()),
        ("GET", format!("/v1/requests/{id}").as_str())
    );
    assert_eq!(calls[3].uri, claims);
    assert_eq!(calls[4].uri, format!("{}/claims/7/ack", route_of(&id)));
    assert_eq!(body_of(&calls[4]), json!({"owner_token": "owner"}));
}

#[test]
fn background_response_can_be_cancelled_once() {
    let processor = StatefulCapturingBackend::new(vec![
        submitted("placeholder"),
        (204, String::new()),
        (200, json!({"id": "x", "status": "queued"}).to_string()),
        (200, json!({"cancelled": 1}).to_string()),
    ])
    .start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config("background_cancel", proxy_port, processor.port());
    let proxy = start_proxy(&config);

    let created = json_of(&http_send(
        proxy.addr(),
        &json_post("/v1/responses", &create_request().to_string()),
    ));
    let id = created["id"].as_str().expect("response id").to_owned();

    for _ in 0..2 {
        let raw = http_send(proxy.addr(), &json_post(&format!("/v1/responses/{id}/cancel"), ""));
        assert_eq!(parse_status(&raw), 200, "{raw}");
        assert_eq!(json_of(&raw)["status"], "cancelled");
    }
    let retrieved = retrieve(proxy.addr(), &id);
    assert_eq!(retrieved["status"], "cancelled");

    let calls = processor.requests();
    assert_eq!(calls.len(), 4, "{:?}", calls.iter().map(|c| &c.uri).collect::<Vec<_>>());
    assert_eq!(
        (calls[3].method.as_str(), calls[3].uri.as_str()),
        ("POST", "/v1/requests/cancel")
    );
    assert_eq!(body_of(&calls[3]), json!({"ids": [id]}));
}

#[test]
fn background_mode_refuses_what_it_does_not_run() {
    let processor = StatefulCapturingBackend::new(Vec::new()).start_with_shutdown();
    let proxy_port = free_port();
    let (config, _db) = load_test_config("background_refused", proxy_port, processor.port());
    let proxy = start_proxy(&config);

    for (extra, message) in [
        (json!({"stream": true}), "background mode does not support stream=true"),
        (
            json!({"conversation": "conv_0"}),
            "background mode does not support conversation",
        ),
        (
            json!({"previous_response_id": "resp_missing"}),
            "previous response 'resp_missing' not found",
        ),
        (
            json!({"tools": [{"type": "web_search"}]}),
            "background mode supports only function tools",
        ),
    ] {
        let mut body = create_request();
        body.as_object_mut()
            .expect("object")
            .extend(extra.as_object().expect("object").clone());
        let raw = http_send(proxy.addr(), &json_post("/v1/responses", &body.to_string()));
        assert_eq!(parse_status(&raw), 400, "{extra}: {raw}");
        assert_eq!(json_of(&raw)["error"]["message"], message, "{extra}");
    }
    assert!(processor.requests().is_empty(), "nothing reaches the processor");
}
