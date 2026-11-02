"""Bring Books to Life — the agent.

A LangGraph ReAct agent (on Groq) that plays a character grounded in an uploaded
book. It decides when to search the book (Agentic RAG), stays in canon, and
REMEMBERS the conversation via a LangGraph checkpointer (thread_id per session).

Needs a free Groq API key as GROQ_API_KEY (see .env.example).
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv, find_dotenv
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    GroqError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from personas import PERSONAS, DEFAULT
from rag import build_retriever, retriever_from_text
from tools import make_search_tool

load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

# openai/gpt-oss-120b does the cleanest structured tool-calling on Groq for this
# chat+retrieval pattern. Tested on this workload, not assumed — re-test if you
# swap models: reliability here is task-specific, not a league table.
# Lighter option: GROQ_MODEL=openai/gpt-oss-20b
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# max_retries: the free Groq tier allows 8000 tokens per MINUTE, and this agent's
# context grows with every tool call — by the later steps a single request is ~4k,
# so two of them inside one minute trips the limit on a single run, with nobody
# else on your key. langchain-groq defaults to 2 retries, which is not enough for
# a long run; the Groq SDK honours the retry-after header, so raising this simply
# waits out the window instead of failing the whole run.
def _llm(temperature: float = 0.3) -> ChatGroq:
    # low-ish temperature: enough character warmth, while keeping tool-calling
    # and the persona stable across turns.
    return ChatGroq(model=MODEL, temperature=temperature, max_retries=8)


def _system(persona) -> str:
    # NOTE: describe the book-search in prose, never as tool_name(args) — that keeps
    # Groq/Llama from emitting a malformed tool call (the Week-1 lesson).
    return (
        f"Your identity is {persona.name}. Speak in the first person, fully in character. "
        "Do not impersonate Sherlock Holmes or any other person mentioned in the source. "
        "The source may describe several characters; use it for relevant facts, but do not "
        "adopt another character's identity, voice, experiences, or habits.\n"
        f"Voice: {persona.voice}\n\n"
        "You have the source book about you. Whenever you need a fact, quote, or detail "
        "about yourself, other characters, or events, look it up in the book before you "
        "answer, so everything you say is true to the book. Do not present unsupported "
        "details as facts; if the book does not cover something, say so in character "
        "rather than making it up.\n\n"
        "Stay in character at all times. Never say you are an AI or a model, and never "
        "mention searching or retrieving — simply speak as yourself. Keep replies natural "
        "and conversational, a few sentences unless asked for more."
    )


class BookChat:
    """One in-character conversation. Remembers previous turns (checkpointer + thread_id)."""

    def __init__(
        self,
        persona_key: str = DEFAULT,
        source_text: str | None = None,
        thread_id: str = "session",
    ):
        self.persona = PERSONAS[persona_key]
        if source_text:
            print(f"RAG source: uploaded text {source_text[:30]!r}", flush=True)
            retriever = retriever_from_text(source_text)
        else:
            print(f"RAG source: file {self.persona.source_path}", flush=True)
            retriever = build_retriever(self.persona.source_path)
        self.agent = create_agent(
            _llm(),
            [make_search_tool(retriever)],
            system_prompt=_system(self.persona),
            checkpointer=InMemorySaver(),
        )
        self.thread_id = thread_id

    def say(self, message: str, verbose: bool = True) -> str:
        try:
            result = self.agent.invoke(
                {"messages": [("user", message)]},
                config={
                    "configurable": {"thread_id": self.thread_id},
                    "recursion_limit": 12,
                },
            )
        except GroqError as exc:
            logger.warning(
                "BookChat model request failed (%s)", type(exc).__name__, exc_info=True
            )
            return _model_error_message(exc)

        messages = result["messages"]
        if verbose:
            _print_trace(messages[-6:])
        return messages[-1].content


def _model_error_message(error: GroqError) -> str:
    """Return a user-safe explanation for a failed Groq model request."""
    if isinstance(error, AuthenticationError):
        return (
            "I can't reach the language model because the Groq API key is invalid or "
            "expired. Update GROQ_API_KEY in your .env file, then restart Bookchat."
        )
    if isinstance(error, RateLimitError):
        return (
            "The language model is temporarily rate-limited. Please wait a moment "
            "before trying again."
        )
    if isinstance(error, BadRequestError):
        details = error.body
        if isinstance(details, dict):
            details = details.get("error", details)
            if isinstance(details, dict):
                details = " ".join(
                    str(details.get(field, ""))
                    for field in ("code", "type", "message", "param")
                )
        details = str(details or error).lower()
        if any(
            marker in details
            for marker in (
                "context_length_exceeded",
                "maximum context length",
                "max context length",
                "too many tokens",
                "token limit",
                "prompt is too long",
                "input is too long",
            )
        ):
            return (
                "This conversation has grown too long for the language model. "
                "Start or reset the conversation, then try a shorter question."
            )
        return (
            "The language model rejected this request. Try a shorter message; if the "
            "problem continues, check the configured model and request settings."
        )
    if isinstance(error, NotFoundError):
        return (
            "The configured language model was not found or is no longer available. "
            "Check the GROQ_MODEL setting."
        )
    if isinstance(error, PermissionDeniedError):
        return (
            "The Groq API key does not have access to the configured model. Check "
            "the key's permissions and the GROQ_MODEL setting."
        )
    if isinstance(error, APITimeoutError):
        return "The language model took too long to respond. Please try again."
    if isinstance(error, APIConnectionError):
        return (
            "I couldn't connect to the language model. Check your internet connection "
            "and try again."
        )
    if isinstance(error, APIStatusError) and error.status_code >= 500:
        return (
            "The language model service is temporarily unavailable. Please try again "
            "in a little while."
        )
    return (
        "The language model request failed unexpectedly. Please try again; if it "
        "continues, check the Groq API key and model configuration."
    )


def _print_trace(messages) -> None:
    for m in messages:
        role = m.__class__.__name__.replace("Message", "").upper()
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            for tc in tcs:
                print(
                    f"  · {role} -> searches the book: {tc['args'].get('query', '')!r}",
                    flush=True,
                )
        elif role == "TOOL":
            snippet = str(m.content).replace("\n", " ")[:110]
            print(f"  · retrieved: {snippet}…", flush=True)
