# chatwoot-support-agent

A support agent for [Chatwoot](https://www.chatwoot.com) built as a **LangGraph state graph** behind a **FastAPI** service. A customer email comes in. The agent classifies it, asks the customer a clarifying question if it's unclear, searches the docs, drafts a reply with citations, and **stops for a human to approve, edit or reject it** before anything is sent.

It reuses the retrieval index from [chatwoot-rag-evals](https://github.com/JohanaG3/chatwoot-rag-evals), where 200-word chunks scored best (recall@5 0.92).

## The graph

```
classify ──support──────────────► search ──useful──► draft ──► review  ⏸ human
   │  └─ambiguous (1st time)─► ask_customer ─► wait_customer ⏸ customer ─► classify
   └─not support / still unclear ─► escalate       │          ├─ approve / edit ─► send
                     search ──nothing useful──► escalate      └─ reject + comment ─► draft (max 3) ─► escalate
```

| Concept | Where |
|---|---|
| Shared state | `Ticket` in [agent/graph.py](agent/graph.py): email, clarifications, chunks, draft, attempts, reviewer feedback, status. Lists are append-only reducers, so history is never overwritten |
| Conditional edges | `after_classify`, `after_search` (similarity threshold), `after_review` (retry budget) |
| Human in the loop | two `interrupt()` pauses: one waits for the customer, one for the reviewer |
| Memory across pauses | SQLite checkpointer: a ticket can wait for days and survive a restart |
| Pausing nodes only wait | a resumed node re-runs from the top, so model calls live in the node *before* each pause |
| Structured output | classification is a pydantic schema; reviewer decisions are validated (reject needs a comment, edit needs text) |

## The service

| Endpoint | |
|---|---|
| `POST /tickets` | a customer email arrives → `202` with `ticket_id` and `trace_id`; the work runs in the background |
| `GET /tickets/{id}` | status, what it's waiting for, and everything the agent remembers |
| `POST /tickets/{id}/customer-reply` | the customer answers the clarifying question |
| `POST /tickets/{id}/decision` | `approve`, `edit` (with your text) or `reject` (with a comment) |
| `GET /tickets/{id}/events` | server-sent events, one per step; ends at the next pause, on `timeout`, or when the client disconnects |

**One trace ID end to end.** The middleware takes the caller's `X-Trace-Id` or W3C `traceparent`, or creates one. The ID travels **inside the job payload** across the queue. The worker restores it into its context, so every JSON log line, every Langfuse span and the outbox record carry it. That holds across the pauses: the run that resumes after an approval days later lands in the same trace.

**Timeouts.** Every model call times out at 30 s, and a whole job at 180 s. Streams stop at their `timeout` and on client disconnect, while the job itself carries on.

## Run it

```bash
uv sync
export GEMINI_API_KEY=...                       # Langfuse keys optional, as in chatwoot-rag-evals
export CHROMA_PATH=../chatwoot-rag-evals/chroma # built by that repo's index.py
uv run uvicorn agent.api:app --port 8480        # then open http://localhost:8480/docs
uv run pytest                                   # 10 end-to-end tests with a fake model, no API calls
```

Sending is simulated: approved replies are appended to `data/outbox.jsonl`.
