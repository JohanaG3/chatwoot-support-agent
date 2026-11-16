"""End-to-end through the HTTP API with the fake model: every route of the graph, both pauses,
the trace id crossing the queue, validation and the SSE stream."""
import json
import os
import time

import pytest

os.environ["AGENT_FAKE_LLM"] = "1"
os.environ["AGENT_NO_TRACING"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

from agent import config  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA", tmp_path)
    from agent.api import app
    with TestClient(app) as cl:
        yield cl


def wait(client, tid, until, timeout=10, attempts=None):
    t = time.time()
    while time.time() - t < timeout:
        r = client.get(f"/tickets/{tid}")
        if r.status_code == 200 and r.json()["status"] in until and attempts in (None, r.json()["attempts"]):
            return r.json()
        time.sleep(0.05)
    raise AssertionError(f"ticket {tid} never reached {until}: {client.get(f'/tickets/{tid}').json()}")


def new(client, subject, body, **headers):
    r = client.post("/tickets", json={"from": "ana@example.com", "subject": subject, "body": body}, headers=headers)
    assert r.status_code == 202
    return r.json()


def test_approve_sends_and_trace_id_survives_the_queue(client, tmp_path):
    t = new(client, "Widget", "¿Cómo pongo el widget en mi web?", **{"X-Trace-Id": "abc123"})
    assert t["trace_id"] == "abc123"
    s = wait(client, t["ticket_id"], {"waiting_approval"})
    assert s["trace_id"] == "abc123" and s["attempts"] == 1 and "Fuentes:" in s["draft"]
    assert client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "approve"}).status_code == 202
    s = wait(client, t["ticket_id"], {"sent"})
    sent = json.loads((tmp_path / "outbox.jsonl").read_text().splitlines()[-1])
    assert sent["trace_id"] == "abc123" and sent["body"] == s["draft"]


def test_traceparent_header_is_accepted(client):
    t = new(client, "Widget", "widget", traceparent="00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
    assert t["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_reject_redrafts_with_feedback_then_escalates_after_three(client):
    t = new(client, "Widget", "widget")
    for i in range(1, 4):
        s = wait(client, t["ticket_id"], {"waiting_approval"}, attempts=i)
        if i > 1:
            assert f"cambio {i - 1}" in s["draft"]
        client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "reject", "comment": f"cambio {i}"})
    s = wait(client, t["ticket_id"], {"escalated"})
    assert s["escalation_reason"] == "3 drafts rejected" and s["feedback"] == ["cambio 1", "cambio 2", "cambio 3"]


def test_edit_waits_for_confirmation_then_sends_the_reviewers_text(client):
    t = new(client, "Widget", "widget")
    wait(client, t["ticket_id"], {"waiting_approval"})
    client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "edit", "text": "Sí, se puede."})
    s = wait(client, t["ticket_id"], {"waiting_confirmation"})
    assert s["waiting_for"]["text"] == "Sí, se puede." and s["final_text"] is None
    client.post(f"/tickets/{t['ticket_id']}/confirm", json={"confirm": True})
    assert wait(client, t["ticket_id"], {"sent"})["final_text"] == "Sí, se puede."


def test_cancelled_edit_goes_back_to_the_same_draft(client):
    t = new(client, "Widget", "widget")
    first = wait(client, t["ticket_id"], {"waiting_approval"})["draft"]
    client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "edit", "text": "Oops"})
    wait(client, t["ticket_id"], {"waiting_confirmation"})
    client.post(f"/tickets/{t['ticket_id']}/confirm", json={"confirm": False})
    s = wait(client, t["ticket_id"], {"waiting_approval"})
    assert s["draft"] == first and s["attempts"] == 1


def test_placeholder_text_is_refused(client):
    t = new(client, "Widget", "widget")
    wait(client, t["ticket_id"], {"waiting_approval"})
    assert client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "edit", "text": "string"}).status_code == 422
    assert client.post(f"/tickets/{t['ticket_id']}/confirm", json={"confirm": True}).status_code == 409


def test_ambiguous_asks_the_customer_then_continues(client):
    t = new(client, "Ayuda", "Necesito ayuda")
    s = wait(client, t["ticket_id"], {"waiting_customer"})
    assert s["waiting_for"]["question"]
    client.post(f"/tickets/{t['ticket_id']}/customer-reply", json={"text": "El chat de mi web"})
    s = wait(client, t["ticket_id"], {"waiting_approval"})
    assert s["clarifications"][0]["answer"] == "El chat de mi web"


def test_still_ambiguous_after_one_question_escalates(client):
    t = new(client, "Ayuda", "Necesito ayuda")
    wait(client, t["ticket_id"], {"waiting_customer"})
    client.post(f"/tickets/{t['ticket_id']}/customer-reply", json={"text": "nada claro"})
    assert wait(client, t["ticket_id"], {"escalated"})["escalation_reason"] == "still unclear after asking the customer"


def test_not_support_and_nothing_found_escalate(client):
    a = new(client, "Oferta", "spam spam")
    b = new(client, "Pago", "¿Puedo pagar en bolívares?")
    assert wait(client, a["ticket_id"], {"escalated"})["escalation_reason"].startswith("not a support question")
    assert wait(client, b["ticket_id"], {"escalated"})["escalation_reason"] == "the documentation has nothing relevant"


def test_wrong_pause_and_bad_decisions_are_refused(client):
    t = new(client, "Widget", "widget")
    wait(client, t["ticket_id"], {"waiting_approval"})
    assert client.post(f"/tickets/{t['ticket_id']}/customer-reply", json={"text": "x"}).status_code == 409
    assert client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "reject"}).status_code == 422
    assert client.post(f"/tickets/{t['ticket_id']}/decision", json={"action": "edit", "text": " "}).status_code == 422
    assert client.get("/tickets/nope").status_code == 404


def test_event_stream_ends_at_the_pause(client):
    t = new(client, "Widget", "widget")
    with client.stream("GET", f"/tickets/{t['ticket_id']}/events?timeout=10") as r:
        body = "".join(r.iter_text())
    events = [l.split(": ", 1)[1] for l in body.splitlines() if l.startswith("event: ")]
    assert events[0] == "job_started" and "step" in events and events[-1] == "waiting"


def test_event_stream_times_out(client):
    t = new(client, "Ayuda", "Necesito ayuda")
    wait(client, t["ticket_id"], {"waiting_customer"})
    with client.stream("GET", f"/tickets/{t['ticket_id']}/events?timeout=0.5&since=99") as r:
        assert "event: timeout" in "".join(r.iter_text())


def test_ticket_id_with_stray_spaces_is_found(client):
    t = new(client, "Ayuda", "Necesito ayuda")
    wait(client, t["ticket_id"], {"waiting_customer"})
    r = client.post(f"/tickets/{t['ticket_id']}%20/customer-reply", json={"text": "El chat de mi web"})
    assert r.status_code == 202
    assert wait(client, t["ticket_id"], {"waiting_approval"})["clarifications"]
