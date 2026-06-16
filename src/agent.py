"""
HR Assistant agent: hybrid query routing + answer synthesis.

Pipeline for each user question:

1. ROUTE   - an LLM classifier labels the query RAG_ONLY, DB_ONLY, or HYBRID
             (with a deterministic keyword fallback if the LLM is unavailable).
2. RETRIEVE- if policy data is needed, fetch top-k chunks from FAISS.
3. TOOLS   - if structured data is needed, let the LLM call database tools.
4. ANSWER  - synthesize a grounded, cited answer from the gathered context.

Routes:
    "What is the maternity leave policy?"              -> RAG_ONLY
    "What is John's leave balance?"                    -> DB_ONLY
    "Maternity policy and how many leaves John has?"   -> HYBRID
"""
from __future__ import annotations

import logging

from src.database import TOOLS, build_tool_executors, run_tool
from src.llm import LLMClient
from src.rag import RAGPipeline

logger = logging.getLogger(__name__)

VALID_ROUTES = {"RAG_ONLY", "DB_ONLY", "HYBRID"}

# Keywords for the deterministic fallback router.
_DB_HINTS = (
    "leave balance", "my leave", "leaves left", "leaves remaining",
    "leaves does", "leave days", "remaining", "days left", "manager",
    "department", "team", "salary", "joining", "my role", "my email",
    "my phone", "who is", "contact",
)
_POLICY_HINTS = (
    "policy", "policies", "benefit", "maternity", "paternity", "onboarding",
    "handbook", "eligibility", "procedure", "process", "entitled", "how do i",
    "how to", "rule",
)

ROUTER_PROMPT = (
    "You are a query router for an HR assistant. Classify the user's question "
    "into exactly one label:\n"
    "- RAG_ONLY: answerable from HR policy documents (leave policy, benefits, "
    "onboarding, handbook rules).\n"
    "- DB_ONLY: answerable from structured employee records (a specific "
    "person's leave balance, manager, department, contact, role).\n"
    "- HYBRID: needs both policy documents AND employee records.\n"
    "Reply with ONLY the label, nothing else."
)

SYNTHESIS_SYSTEM = (
    "You are HRAssistant, a helpful and precise HR copilot. Answer the "
    "employee's question using ONLY the provided context (policy excerpts and "
    "database results). Be concise and professional. If the context does not "
    "contain the answer, say so clearly and do not invent facts. Do not add a "
    "Sources line yourself; it is appended automatically."
)


class HRAgent:
    def __init__(self, llm: LLMClient, rag: RAGPipeline, db):
        self.llm = llm
        self.rag = rag
        self.db = db
        self.executors = build_tool_executors(db)

    # --- Routing --------------------------------------------------------------
    def route(self, query: str) -> str:
        """Classify the query into RAG_ONLY / DB_ONLY / HYBRID."""
        try:
            label = self.llm.chat(
                [
                    {"role": "system", "content": ROUTER_PROMPT},
                    {"role": "user", "content": query},
                ]
            ).strip().upper()
            for route in VALID_ROUTES:
                if route in label:
                    return route
        except Exception as e:  # noqa: BLE001 - fall back to keywords
            logger.warning("LLM router failed, using keyword fallback: %s", e)
        return self._keyword_route(query)

    @staticmethod
    def _keyword_route(query: str) -> str:
        q = query.lower()
        wants_db = any(h in q for h in _DB_HINTS)
        wants_policy = any(h in q for h in _POLICY_HINTS)
        if wants_db and wants_policy:
            return "HYBRID"
        if wants_db:
            return "DB_ONLY"
        return "RAG_ONLY"  # default: most questions are policy questions

    # --- Data gathering -------------------------------------------------------
    def _gather_db_context(self, query: str, emp_id: str | None) -> str:
        """Let the LLM call database tools and collect their results.

        The router has already decided structured data is needed, so we ask the
        model to call at least one tool (tool_choice="required"). Some free
        models don't support "required"; if that call fails we retry with
        "auto".
        """
        hint = f"\nThe current employee's ID is {emp_id}." if emp_id else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "Call the available tools to fetch the HR data needed to "
                    "answer the user's question. Resolve 'me'/'my'/'I' to the "
                    "current employee when an ID is given. Always call a tool "
                    "rather than answering from memory." + hint
                ),
            },
            {"role": "user", "content": query},
        ]
        try:
            message = self.llm.chat_with_tools(messages, TOOLS, tool_choice="required")
        except Exception:  # noqa: BLE001 - model may not support forced tools
            message = self.llm.chat_with_tools(messages, TOOLS, tool_choice="auto")

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return ""

        results = []
        for call in tool_calls:
            result = run_tool(self.executors, call.function.name, call.function.arguments)
            results.append(f"{call.function.name}({call.function.arguments}) -> {result}")
        return "\n".join(results)

    # --- Main entry point -----------------------------------------------------
    def answer(self, query: str, emp_id: str | None = None, history: list[dict] | None = None) -> dict:
        """Route, gather context, and synthesize a cited answer."""
        route = self.route(query)

        policy_chunks: list[dict] = []
        policy_context = ""
        db_context = ""
        sources: list[str] = []

        if route in ("RAG_ONLY", "HYBRID"):
            policy_chunks = self.rag.retrieve(query)
            policy_context = self.rag.format_context(policy_chunks)
            sources = self.rag.sources(policy_chunks)

        if route in ("DB_ONLY", "HYBRID"):
            db_context = self._gather_db_context(query, emp_id)

        answer_text = self._synthesize(query, policy_context, db_context, history)

        citations = []
        if sources:
            citations.append("Policy documents: " + ", ".join(sources))
        if db_context:
            citations.append("Employee database")
        if citations:
            answer_text += "\n\n---\n*Sources: " + "; ".join(citations) + "*"

        return {"answer": answer_text, "route": route, "sources": sources}

    def _synthesize(
        self,
        query: str,
        policy_context: str,
        db_context: str,
        history: list[dict] | None,
    ) -> str:
        context_blocks = []
        if policy_context:
            context_blocks.append("HR POLICY EXCERPTS:\n" + policy_context)
        if db_context:
            context_blocks.append("EMPLOYEE DATABASE RESULTS:\n" + db_context)
        context = "\n\n".join(context_blocks) if context_blocks else "(no context retrieved)"

        messages = [{"role": "system", "content": SYNTHESIS_SYSTEM}]
        for msg in (history or [])[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }
        )
        return self.llm.chat(messages)
