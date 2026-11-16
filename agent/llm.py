"""The three model calls and the retrieval call. Kept behind one small interface so tests can swap
them for a deterministic fake (see tests/fake.py)."""
import time
from typing import Literal

from pydantic import BaseModel, Field

from . import config as c


class Classification(BaseModel):
    category: Literal["support", "ambiguous", "not_support"] = Field(
        description="support: a question about using Chatwoot that the docs could answer; "
                    "ambiguous: probably about Chatwoot but it's unclear what they need; "
                    "not_support: spam, sales, billing disputes, or anything else")
    reason: str = Field(description="One sentence")


CLASSIFY = """Clasificas correos que llegan al soporte de Chatwoot (software de atención al cliente).
- support: una pregunta sobre cómo usar o instalar Chatwoot que la documentación podría responder.
- ambiguous: parece sobre Chatwoot, pero no queda claro qué necesita (falta qué función, qué canal o qué error).
- not_support: spam, ventas, reclamos de facturación, temas personales o cualquier otra cosa.
Si el cliente ya aclaró algo, tenlo en cuenta."""

CLARIFY = """Eres el soporte de Chatwoot. El correo del cliente no deja claro qué necesita.
Escribe UNA sola pregunta breve y amable, en español, que te permita saber qué función o problema tiene.
Devuelve solo la pregunta."""

DRAFT = """Eres el soporte de Chatwoot y redactas una respuesta por correo, en español, de 3 a 6 frases, cálida y concreta.
Usa la información de los fragmentos, pero úsalos todos: evalúa el contexto que tiene cada uno, a ver si encuentras la respuesta.
Cita cada afirmación con el número del fragmento, por ejemplo [2]. Los saludos y despedidas no llevan cita.
Si no está claro o definitivamente no hay nada, no lo inventes: dilo y ofrece pasar el caso a una persona del equipo.
Si sí encontraste la respuesta, no ofrezcas pasar el caso a una persona.
Si hay comentarios del revisor, corrígelos todos en esta versión y que el cambio se note:
si pide algo más corto, usa como mucho la mitad de frases que el borrador anterior (máximo 3)."""


class Gemini:
    def __init__(self):
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=c.secret("GEMINI_API_KEY", ".gemini_key"),
                                   http_options=types.HttpOptions(timeout=c.LLM_TIMEOUT_MS))
        import chromadb
        self.collection = chromadb.PersistentClient(path=str(c.CHROMA)).get_collection(c.COLLECTION)

    def _retry(self, fn, attempts=6):
        for i in range(attempts):
            try:
                return fn()
            except Exception as e:  # free tier: 429 and 503 are routine
                if i == attempts - 1:
                    raise
                c.info("llm.retry", error=str(e)[:120], attempt=i + 1)
                time.sleep(5 * (i + 1))

    def _generate(self, name, system, message, schema=None):
        cfg = self.types.GenerateContentConfig(system_instruction=system, temperature=0)
        if schema:
            cfg.response_mime_type, cfg.response_schema = "application/json", schema
        with c.observe(name, as_type="generation", model=c.LLM_MODEL, input=message) as gen:
            t = time.monotonic()
            resp = self._retry(lambda: self.client.models.generate_content(model=c.LLM_MODEL, contents=message, config=cfg))
            u = resp.usage_metadata
            gen.update(output=resp.text, usage_details={"input": u.prompt_token_count or 0,
                                                       "output": u.candidates_token_count or 0})
            c.info("llm.call", step=name, ms=int((time.monotonic() - t) * 1000),
                   tokens_in=u.prompt_token_count, tokens_out=u.candidates_token_count)
        return resp

    def classify(self, email_text):
        resp = self._generate("classify", CLASSIFY, email_text, schema=Classification)
        return Classification.model_validate_json(resp.text)

    def clarifying_question(self, email_text):
        return self._generate("clarify", CLARIFY, email_text).text.strip()

    def draft(self, email_text, context, feedback, previous=None):
        notes = "\n".join(f"- {f}" for f in feedback)
        message = f"Fragmentos:\n{context}\n\n{email_text}"
        if previous:
            message += f"\n\nBorrador anterior (rechazado):\n{previous}"
        if notes:
            message += f"\n\nComentarios del revisor:\n{notes}"
        return self._generate("draft", DRAFT, message).text.strip()

    def retrieve(self, query):
        with c.observe("retrieve", as_type="retriever", input=query) as step:
            qv = self._retry(lambda: self.client.models.embed_content(
                model=c.EMBED_MODEL, contents=query,
                config=self.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=c.EMBED_DIM))
            ).embeddings[0].values
            r = self.collection.query(query_embeddings=[qv], n_results=c.TOP_K)
            chunks = [{"n": i + 1, "title": m["title"], "url": m["url"], "similarity": round(1 - d, 3), "text": doc}
                      for i, (doc, m, d) in enumerate(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))]
            step.update(output=[{k: x[k] for k in ("n", "title", "similarity")} for x in chunks])
        return chunks
