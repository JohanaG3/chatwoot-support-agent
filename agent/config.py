"""Settings, clients, structured logging and tracing helpers shared by the graph and the API."""
import contextlib
import contextvars
import json
import logging
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
CHROMA = pathlib.Path(os.environ.get("CHROMA_PATH", ROOT.parent / "chatwoot-rag-evals" / "chroma"))
COLLECTION = os.environ.get("CHROMA_COLLECTION", "chatwoot_200")  # 200-word chunks retrieved better (recall@5 0.92)

EMBED_MODEL, EMBED_DIM = "gemini-embedding-001", 768
LLM_MODEL = "gemini-3.5-flash-lite"
LLM_TIMEOUT_MS = 30_000        # every model call gives up after 30 s instead of hanging a worker
TOP_K = 5
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.62"))  # below this, search "found nothing useful"
MAX_CLARIFICATIONS = 1         # questions to the customer before giving up and escalating
MAX_DRAFTS = 3                 # drafts a reviewer can reject before the ticket goes to a person


def secret(env, file):
    if os.environ.get(env):
        return os.environ[env].strip()
    path = pathlib.Path.home() / file
    return path.read_text().strip() if path.exists() else None


# --- correlation ids: set once per request or job, read by every log line and trace -------------
trace_id_var = contextvars.ContextVar("trace_id", default=None)
ticket_id_var = contextvars.ContextVar("ticket_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        line = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
                "level": record.levelname.lower(), "msg": record.getMessage(),
                "trace_id": trace_id_var.get(), "ticket_id": ticket_id_var.get()}
        line.update(getattr(record, "fields", {}))
        return json.dumps(line, ensure_ascii=False)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(JsonFormatter())
log = logging.getLogger("agent")
log.handlers[:] = [_handler]
log.setLevel(logging.INFO)
log.propagate = False


def info(msg, **fields):
    log.info(msg, extra={"fields": fields})


# --- Langfuse (optional): every segment of a ticket lands in the same trace ----------------------
_pk, _sk = secret("LANGFUSE_PUBLIC_KEY", ".langfuse_public"), secret("LANGFUSE_SECRET_KEY", ".langfuse_secret")
lf = None
if _pk and _sk and not os.environ.get("AGENT_NO_TRACING"):
    from langfuse import Langfuse
    lf = Langfuse(public_key=_pk, secret_key=_sk, host=os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com"))


class _NoOp:
    def update(self, **_):
        pass


@contextlib.contextmanager
def observe(name, **kwargs):
    """A Langfuse observation inside the current ticket's trace, or a no-op without keys."""
    if lf is None:
        yield _NoOp()
        return
    tid = trace_id_var.get()
    if tid and "trace_context" not in kwargs:
        kwargs["trace_context"] = {"trace_id": tid}
    with lf.start_as_current_observation(name=name, **kwargs) as o:
        yield o
