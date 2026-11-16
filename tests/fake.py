"""Deterministic stand-in for the model and the search, so graph routing can be tested for free.

Keywords in the email decide the path:  "spam" -> not_support,  "ayuda" alone -> ambiguous
(until the customer clarifies),  "bolívares" -> search finds nothing useful.
"""
from agent.llm import Classification


class FakeLLM:
    def __init__(self):
        self.drafts = 0

    def classify(self, text):
        low = text.lower()
        if "spam" in low:
            return Classification(category="not_support", reason="spam")
        if "nada claro" in low:
            return Classification(category="ambiguous", reason="still unclear")
        if "respuesta del cliente" in low or "widget" in low or "bolívares" in low:
            return Classification(category="support", reason="clear question")
        return Classification(category="ambiguous", reason="unclear")

    def clarifying_question(self, text):
        return "¿Qué función de Chatwoot quieres usar?"

    def draft(self, text, context, feedback):
        self.drafts += 1
        return f"Borrador {self.drafts}: pega el script del widget [1]." + (f" (corrige: {feedback[-1]})" if feedback else "")

    def retrieve(self, query):
        sim = 0.3 if "bolívares" in query.lower() else 0.8
        return [{"n": 1, "title": "Website live chat settings explained", "url": "https://example.com/widget",
                 "similarity": sim, "text": "Copia el script del widget en tu web."}]
