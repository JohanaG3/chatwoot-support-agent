"""HTTP API around the graph.

Request path:  client ─► middleware (trace_id from X-Trace-Id / traceparent, or a new one)
                      ─► endpoint (validates, enqueues a job that carries the trace_id) ─► 202
Worker path:   queue ─► worker (restores trace_id + ticket_id into its context) ─► graph run
                      ─► events for the SSE stream, JSON logs and Langfuse spans, all with the same trace_id

Run: uv run uvicorn agent.api:app --port 8480
"""
import asyncio
import contextlib
import functools
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from . import config as c
from .graph import build

JOB_TIMEOUT_S = 180


@dataclass
class Job:
    ticket_id: str
    trace_id: str
    payload: object          # the email for a new ticket, or the resume value for a paused one
    resume: bool


class EmailIn(BaseModel):
    from_: str = Field(alias="from", examples=["ana@example.com"])
    subject: str = Field(examples=["Widget en mi web"])
    body: str = Field(examples=["Hola, ¿cómo pongo el chat de Chatwoot en mi página de WordPress?"])


class ReplyIn(BaseModel):
    text: str = Field(examples=["Me refiero al chat en vivo de mi página web."])


class DecisionIn(BaseModel):
    action: Literal["approve", "edit", "reject"]
    text: str | None = Field(None, description="Your version of the reply (action=edit)")
    comment: str | None = Field(None, description="What to change (action=reject)")

    @model_validator(mode="after")
    def check(self):
        if self.action == "edit" and not (self.text or "").strip():
            raise ValueError("edit needs text")
        if self.action == "reject" and not (self.comment or "").strip():
            raise ValueError("reject needs a comment")
        return self


def make_llm():
    if os.environ.get("AGENT_FAKE_LLM"):
        from tests.fake import FakeLLM
        return FakeLLM()
    from .llm import Gemini
    return Gemini()


class Events:
    """Per-ticket event log for the SSE stream (in memory: a demo, not a durable bus).
    Only touched from the event loop thread; the worker thread hands events over with call_soon_threadsafe."""

    def __init__(self):
        self.items, self.changed = {}, asyncio.Event()

    def add(self, _ticket_id, _event, /, **data):
        log = self.items.setdefault(_ticket_id, [])
        log.append({**data, "seq": len(log), "event": _event, "ts": time.time()})
        self.changed.set()
        self.changed = asyncio.Event()


@contextlib.asynccontextmanager
async def lifespan(app):
    conn = sqlite3.connect(c.DATA / "checkpoints.db", check_same_thread=False)
    app.state.graph = build(make_llm(), SqliteSaver(conn))
    app.state.queue = asyncio.Queue()
    app.state.events = Events()
    worker = asyncio.create_task(run_worker(app))
    c.info("api.start", collection=c.COLLECTION)
    yield
    with contextlib.suppress(asyncio.TimeoutError):  # let a running job finish before closing the database
        await asyncio.wait_for(app.state.queue.join(), 10)
    worker.cancel()
    if c.lf:
        c.lf.flush()
    conn.close()


app = FastAPI(title="Chatwoot support agent", lifespan=lifespan,
              description="LangGraph support agent with a customer-clarification pause and a human-approval pause.")


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    """Ingress: accept the caller's trace id (X-Trace-Id or W3C traceparent) or start a new one."""
    tid = request.headers.get("x-trace-id")
    if not tid and (tp := request.headers.get("traceparent")):
        parts = tp.split("-")
        tid = parts[1] if len(parts) == 4 else None
    tid = tid or uuid.uuid4().hex
    token = c.trace_id_var.set(tid)
    t = time.monotonic()
    try:
        response = await call_next(request)
        response.headers["X-Trace-Id"] = tid
        c.info("http", method=request.method, path=request.url.path, status=response.status_code,
               ms=int((time.monotonic() - t) * 1000))
        return response
    finally:
        c.trace_id_var.reset(token)


# --- worker: the other side of the queue ------------------------------------------------------
async def run_worker(app):
    while True:
        job = await app.state.queue.get()
        c.trace_id_var.set(job.trace_id)       # the ids crossed the queue inside the job payload
        c.ticket_id_var.set(job.ticket_id)
        c.info("job.start", resume=job.resume)
        app.state.events.add(job.ticket_id, "job_started", resume=job.resume)
        try:
            await asyncio.wait_for(asyncio.to_thread(run_graph, app, job, asyncio.get_running_loop()), JOB_TIMEOUT_S)
        except asyncio.TimeoutError:
            # The thread can't be killed, but every model call has its own 30 s timeout, so it ends soon.
            c.info("job.timeout", seconds=JOB_TIMEOUT_S)
            app.state.events.add(job.ticket_id, "error", detail=f"job exceeded {JOB_TIMEOUT_S}s")
        except Exception as e:
            c.info("job.error", error=str(e)[:200])
            app.state.events.add(job.ticket_id, "error", detail=str(e)[:200])
        finally:
            app.state.queue.task_done()
            c.ticket_id_var.set(None)


def run_graph(app, job, loop):
    graph, cfg = app.state.graph, {"configurable": {"thread_id": job.ticket_id}}
    emit = lambda _event, **d: loop.call_soon_threadsafe(functools.partial(app.state.events.add, job.ticket_id, _event, **d))
    graph_input = Command(resume=job.payload) if job.resume else {
        "ticket_id": job.ticket_id, "trace_id": job.trace_id, "email": job.payload, "attempts": 0}
    with c.observe("support-ticket", as_type="chain", input=job.payload,
                   metadata={"ticket_id": job.ticket_id, "resume": job.resume}) as root:
        for update in graph.stream(graph_input, cfg, stream_mode="updates"):
            for name, out in update.items():
                if name == "__interrupt__":
                    emit("waiting", **out[0].value)
                else:
                    emit("step", node=name, **{k: v for k, v in (out or {}).items() if k != "chunks"})
        snap = graph.get_state(cfg)
        root.update(output={"status": snapshot_status(snap)})
    emit("status", status=snapshot_status(snap))


# --- helpers ------------------------------------------------------------------------------------
def pending(snap):
    for task in snap.tasks:
        for it in task.interrupts:
            return it.value
    return None


def snapshot_status(snap):
    if not snap.values:
        return "unknown"
    if p := pending(snap):
        return "waiting_customer" if p["kind"] == "customer_reply" else "waiting_approval"
    return snap.values.get("status") or "running"


def get_snapshot(request, ticket_id):
    snap = request.app.state.graph.get_state({"configurable": {"thread_id": ticket_id}})
    if not snap.values:
        raise HTTPException(404, "unknown ticket")
    return snap


async def enqueue(request, ticket_id, trace_id, payload, resume):
    c.ticket_id_var.set(ticket_id)
    c.info("job.enqueued", resume=resume)
    await request.app.state.queue.put(Job(ticket_id, trace_id, payload, resume))


# --- endpoints ----------------------------------------------------------------------------------
@app.post("/tickets", status_code=202)
async def create_ticket(email: EmailIn, request: Request):
    """A customer email arrives. Returns at once; the agent works in the background."""
    ticket_id, tid = uuid.uuid4().hex[:8], c.trace_id_var.get()
    request.app.state.events.add(ticket_id, "received")
    await enqueue(request, ticket_id, tid, email.model_dump(by_alias=True), resume=False)
    return {"ticket_id": ticket_id, "trace_id": tid, "status_url": f"/tickets/{ticket_id}",
            "events_url": f"/tickets/{ticket_id}/events"}


@app.get("/tickets")
def list_tickets(request: Request):
    saver = request.app.state.graph.checkpointer
    ids = list(dict.fromkeys(cp.config["configurable"]["thread_id"] for cp in saver.list(None)))
    out = []
    for tid in ids[:50]:
        snap = request.app.state.graph.get_state({"configurable": {"thread_id": tid}})
        out.append({"ticket_id": tid, "subject": snap.values.get("email", {}).get("subject"),
                    "status": snapshot_status(snap)})
    return out


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str, request: Request):
    """Where the ticket is, what the agent remembers, and what it is waiting for."""
    snap = get_snapshot(request, ticket_id)
    v = snap.values
    return {"ticket_id": ticket_id, "trace_id": v.get("trace_id"), "status": snapshot_status(snap),
            "waiting_for": pending(snap), "next": list(snap.next), "category": v.get("category"),
            "clarifications": v.get("clarifications", []), "attempts": v.get("attempts", 0),
            "feedback": v.get("feedback", []), "draft": v.get("draft"), "final_text": v.get("final_text"),
            "escalation_reason": v.get("escalation_reason"),
            "sources": [{k: x[k] for k in ("n", "title", "similarity")} for x in v.get("chunks", [])]}


@app.post("/tickets/{ticket_id}/customer-reply", status_code=202)
async def customer_reply(ticket_id: str, reply: ReplyIn, request: Request):
    """The customer answers the clarifying question."""
    snap = get_snapshot(request, ticket_id)
    if (pending(snap) or {}).get("kind") != "customer_reply":
        raise HTTPException(409, f"ticket is {snapshot_status(snap)}, not waiting for the customer")
    await enqueue(request, ticket_id, snap.values["trace_id"], reply.text, resume=True)
    return {"ticket_id": ticket_id, "queued": "customer_reply"}


@app.post("/tickets/{ticket_id}/decision", status_code=202)
async def decision(ticket_id: str, d: DecisionIn, request: Request):
    """Approve the draft, send your own edited version, or reject it with a comment."""
    snap = get_snapshot(request, ticket_id)
    if (pending(snap) or {}).get("kind") != "approval":
        raise HTTPException(409, f"ticket is {snapshot_status(snap)}, not waiting for approval")
    await enqueue(request, ticket_id, snap.values["trace_id"], d.model_dump(), resume=True)
    return {"ticket_id": ticket_id, "queued": d.action}


@app.get("/tickets/{ticket_id}/events")
async def events(ticket_id: str, request: Request, timeout: float = 60, since: int | None = None):
    """Server-sent events: each step as it happens. Ends when the ticket pauses or finishes,
    after `timeout` seconds, or as soon as the client disconnects (the work itself carries on).
    By default it starts at the latest run of the ticket; pass `since` to replay from an event number."""
    ev = request.app.state.events
    if since is None:
        starts = [e["seq"] for e in ev.items.get(ticket_id, []) if e["event"] in ("received", "job_started")]
        since = starts[-1] if starts else 0
    deadline = time.monotonic() + min(timeout, 300)

    async def stream():
        i = since
        while True:
            changed = ev.changed
            if await request.is_disconnected():
                c.info("sse.client_disconnected", ticket_id=ticket_id)
                return
            items = ev.items.get(ticket_id, [])
            while i < len(items):
                e = items[i]
                i += 1
                yield f"event: {e['event']}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n"
                if e["event"] in ("waiting", "error") or (e["event"] == "status" and e["status"] in ("sent", "escalated")):
                    return
            left = deadline - time.monotonic()
            if left <= 0:
                yield f"event: timeout\ndata: {json.dumps({'next_since': i})}\n\n"
                return
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(changed.wait(), min(left, 1.0))

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
