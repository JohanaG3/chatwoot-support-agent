"""The support agent as a LangGraph state graph.

    classify ──support──────────────► search ──useful──► draft ──► review (pause: human)
       │  └─ambiguous (1st time)─► ask_customer ─► wait_customer (pause: customer) ─► classify
       └─not_support / still ambiguous ─► escalate        ├─ approve ─► send
                            search ──nothing useful──► escalate ├─ edit ─► confirm_edit (pause) ─► send / back to review
                                                          └─ reject ─► draft (max 3) ─► escalate

Pauses use interrupt(): the graph stops, the checkpointer saves the state, and the run resumes when
the API receives the customer's reply or the reviewer's decision, minutes or days later.
Nodes that call a model never pause: a resumed node re-runs from the top, so pausing nodes only wait.
"""
import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import config as c


class Ticket(TypedDict, total=False):
    ticket_id: str
    trace_id: str
    email: dict                                   # {"from", "subject", "body"}
    category: str
    category_reason: str
    pending_question: str
    clarifications: Annotated[list, operator.add]  # [{"question", "answer"}], appended, never overwritten
    chunks: list
    draft: str
    attempts: int
    feedback: Annotated[list, operator.add]        # reviewer comments, appended
    decision: str
    edited_text: str
    final_text: str
    status: str                                    # sent | escalated
    escalation_reason: str


def email_text(t):
    e = t["email"]
    text = f"De: {e.get('from', '')}\nAsunto: {e.get('subject', '')}\n\n{e.get('body', '')}"
    for cl in t.get("clarifications", []):
        text += f"\n\nPregunta de soporte: {cl['question']}\nRespuesta del cliente: {cl['answer']}"
    return text


def context_text(chunks):
    return "\n\n".join(f"[{x['n']}] ({x['title']})\n{x['text']}" for x in chunks)


def sources(draft, chunks):
    """Append the cited articles' links, so the customer can check every claim."""
    cited = {x["n"] for x in chunks if f"[{x['n']}" in draft or f", {x['n']}" in draft}
    by_url = {}
    for x in chunks:
        if x["n"] in cited:
            by_url.setdefault(x["url"], (x["title"], []))[1].append(str(x["n"]))
    lines = [f"[{', '.join(ns)}] {title}: {url}" for url, (title, ns) in by_url.items()]
    return draft + ("\n\nFuentes:\n" + "\n".join(lines) if lines else "")


def build(llm, checkpointer):
    def node(name):
        """Each node becomes a span in the ticket's trace and a structured log line."""
        def wrap(fn):
            def run(t):
                c.ticket_id_var.set(t.get("ticket_id"))
                c.trace_id_var.set(t.get("trace_id"))
                with c.observe(name, as_type="span") as span:
                    out = fn(t)
                    span.update(output={k: v for k, v in out.items() if k not in ("chunks",)})
                c.info("node.done", node=name, **{k: v for k, v in out.items() if isinstance(v, (str, int)) and len(str(v)) < 120})
                return out
            run.__name__ = name
            return run
        return wrap

    @node("classify")
    def classify(t):
        r = llm.classify(email_text(t))
        return {"category": r.category, "category_reason": r.reason}

    @node("ask_customer")
    def ask_customer(t):
        return {"pending_question": llm.clarifying_question(email_text(t))}

    @node("wait_customer")
    def wait_customer(t):
        answer = interrupt({"kind": "customer_reply", "question": t["pending_question"]})
        return {"clarifications": [{"question": t["pending_question"], "answer": answer}], "pending_question": ""}

    @node("search")
    def search(t):
        e = t["email"]
        query = " ".join([e.get("subject", ""), e.get("body", "")] + [x["answer"] for x in t.get("clarifications", [])])
        return {"chunks": llm.retrieve(query)}

    @node("draft")
    def draft(t):
        previous = t.get("draft", "").split("\n\nFuentes:")[0] if t.get("feedback") else None
        text = llm.draft(email_text(t), context_text(t["chunks"]), t.get("feedback", []), previous)
        return {"draft": sources(text, t["chunks"]), "attempts": t.get("attempts", 0) + 1}

    @node("review")
    def review(t):
        d = interrupt({"kind": "approval", "draft": t["draft"], "attempt": t["attempts"], "max_attempts": c.MAX_DRAFTS})
        if d["action"] == "approve":
            return {"decision": "approve", "final_text": t["draft"]}
        if d["action"] == "edit":
            return {"decision": "edit", "edited_text": d["text"]}
        return {"decision": "reject", "feedback": [d["comment"]]}

    @node("confirm_edit")
    def confirm_edit(t):
        """A human-written reply goes out verbatim, so the reviewer sees it once more before it's sent."""
        ok = interrupt({"kind": "confirm_edit", "text": t["edited_text"]})
        if ok:
            return {"decision": "edit_confirmed", "final_text": t["edited_text"]}
        return {"decision": "edit_cancelled", "edited_text": ""}

    @node("send")
    def send(t):
        with open(c.DATA / "outbox.jsonl", "a") as f:  # stands in for the Chatwoot API / SMTP
            f.write(json.dumps({"ticket_id": t["ticket_id"], "trace_id": t["trace_id"], "to": t["email"].get("from"),
                                "subject": "Re: " + t["email"].get("subject", ""), "body": t["final_text"]},
                               ensure_ascii=False) + "\n")
        return {"status": "sent"}

    @node("escalate")
    def escalate(t):
        if t.get("category") == "not_support":
            reason = f"not a support question: {t.get('category_reason', '')}"
        elif t.get("category") == "ambiguous":
            reason = "still unclear after asking the customer"
        elif t.get("attempts", 0) >= c.MAX_DRAFTS:
            reason = f"{c.MAX_DRAFTS} drafts rejected"
        else:
            reason = "the documentation has nothing relevant"
        return {"status": "escalated", "escalation_reason": reason}

    def after_classify(t):
        if t["category"] == "support":
            return "search"
        if t["category"] == "ambiguous" and len(t.get("clarifications", [])) < c.MAX_CLARIFICATIONS:
            return "ask_customer"
        return "escalate"

    def after_search(t):
        best = max((x["similarity"] for x in t["chunks"]), default=0)
        return "draft" if best >= c.MIN_SIMILARITY else "escalate"

    def after_review(t):
        if t["decision"] == "approve":
            return "send"
        if t["decision"] == "edit":
            return "confirm_edit"
        return "draft" if t["attempts"] < c.MAX_DRAFTS else "escalate"

    g = StateGraph(Ticket)
    def after_confirm(t):
        return "send" if t["decision"] == "edit_confirmed" else "review"

    for fn in (classify, ask_customer, wait_customer, search, draft, review, confirm_edit, send, escalate):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", after_classify, ["search", "ask_customer", "escalate"])
    g.add_edge("ask_customer", "wait_customer")
    g.add_edge("wait_customer", "classify")
    g.add_conditional_edges("search", after_search, ["draft", "escalate"])
    g.add_edge("draft", "review")
    g.add_conditional_edges("review", after_review, ["send", "confirm_edit", "draft", "escalate"])
    g.add_conditional_edges("confirm_edit", after_confirm, ["send", "review"])
    g.add_edge("send", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer)
