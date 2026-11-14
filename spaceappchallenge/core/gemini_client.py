"""
Chat API client wrapper for the Flame in Freefall application.
Provides text generation via OpenRouter (OpenAI-compatible) or Google Gemini,
with an intelligent offline grounding fallback.
"""

import json
import logging
from typing import Any, Generator, Optional

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from config import (
    CHAT_MAX_TOKENS,
    EMBEDDING_MODEL,
    FALLBACK_MAX_TOKENS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_CHAT_MODEL,
    OPENROUTER_FALLBACK_MODEL,
    OPENROUTER_FALLBACK_MODEL_2,
    OPENROUTER_FALLBACK_MODEL_3,
    SYSTEM_PROMPT,
    is_openrouter_key,
)

logger = logging.getLogger(__name__)


class GeminiClient:
    """Chat client supporting Google Gemini and OpenRouter (OpenAI-compatible) APIs."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or GEMINI_API_KEY
        self._configured = False
        self._model = None
        self._embedding_model_name = EMBEDDING_MODEL
        self._provider = "offline"
        self._groq_enabled = bool(GROQ_API_KEY) and HAS_REQUESTS

        if is_openrouter_key(self.api_key) and HAS_REQUESTS:
            self._configured = True
            self._provider = "openrouter"
            logger.info("Chat client configured via OpenRouter using model %s", OPENROUTER_CHAT_MODEL)
            if self._groq_enabled:
                logger.info("Groq fallback enabled using model %s", GROQ_MODEL)
        elif self.api_key and self.api_key != "your_gemini_api_key_here":
            self._provider = "google"
            if HAS_GENAI:
                try:
                    genai.configure(api_key=self.api_key)
                    self._model = genai.GenerativeModel(GEMINI_MODEL)
                    self._configured = True
                    logger.info("Gemini client configured with SDK using model %s", GEMINI_MODEL)
                except Exception as e:
                    logger.warning("SDK configuration failed: %s; will use REST if available", e)
                    self._configured = HAS_REQUESTS
            elif HAS_REQUESTS:
                self._configured = True
                logger.info("Gemini client configured with REST API using model %s", GEMINI_MODEL)
            else:
                self._configured = False
            if self._groq_enabled:
                logger.info("Groq fallback enabled using model %s", GROQ_MODEL)
        elif self._groq_enabled:
            self._configured = True
            self._provider = "groq"
            logger.info("Chat client configured via Groq using model %s", GROQ_MODEL)
        else:
            logger.info("Running in offline/local knowledge mode (no API key set).")
            self._configured = False

    @property
    def is_configured(self) -> bool:
        """Check whether the chat client has an active API key."""
        return self._configured

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def provider_label(self) -> str:
        if self._provider == "openrouter":
            label = f"OpenRouter ({OPENROUTER_CHAT_MODEL})"
            if self._groq_enabled:
                label += f" + Groq fallback ({GROQ_MODEL})"
            return label
        if self._provider == "groq":
            return f"Groq ({GROQ_MODEL})"
        if self._provider == "google":
            label = f"Google Gemini ({GEMINI_MODEL})"
            if self._groq_enabled:
                label += f" + Groq fallback ({GROQ_MODEL})"
            return label
        return "Local Knowledge Grounding"

    # ── Text Generation ───────────────────────────────────────────────────────

    def generate_response(
        self, prompt: str, context: Optional[str] = None, stream: bool = True
    ) -> str | Generator[str, None, None]:
        """Generate a response grounded in microgravity combustion data."""
        # 1. OpenRouter (OpenAI-compatible), with Groq as secondary fallback
        if self._provider == "openrouter" and self._configured:
            if stream:
                return self._stream_with_fallbacks(prompt, context, ["openrouter", "groq"])
            try:
                return self._call_openrouter(prompt, context)
            except Exception as e:
                logger.error("OpenRouter generation error: %s", e)
            if self._groq_enabled:
                try:
                    return self._call_groq(prompt, context)
                except Exception as e:
                    logger.error("Groq generation error: %s", e)

        # 2. Groq-only configuration
        elif self._provider == "groq" and self._groq_enabled:
            if stream:
                return self._stream_with_fallbacks(prompt, context, ["groq"])
            try:
                return self._call_groq(prompt, context)
            except Exception as e:
                logger.error("Groq generation error: %s", e)

        full_prompt = self._build_prompt(prompt, context)

        # 3. Google Gemini, with Groq as a secondary fallback
        if self._configured and self._provider == "google":
            if stream:
                return self._stream_with_fallbacks(prompt, context, ["google", "groq"])

            if HAS_GENAI and self._model is not None:
                try:
                    response = self._model.generate_content(full_prompt)
                    return response.text
                except Exception as e:
                    logger.warning("SDK generation error (%s), trying REST fallback", e)

            if HAS_REQUESTS and self.api_key:
                try:
                    return self._call_rest_api(full_prompt)
                except Exception as e:
                    logger.error("REST generation error: %s", e)

            if self._groq_enabled:
                try:
                    return self._call_groq(prompt, context)
                except Exception as e:
                    logger.error("Groq generation error: %s", e)
        offline_ans = self._generate_offline_grounded_response(prompt, context)
        if stream:
            return (chunk for chunk in [offline_ans])
        return offline_ans

    def _chat_messages(self, prompt: str, context: Optional[str]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(prompt, context)},
        ]

    @staticmethod
    def _openrouter_candidates() -> list[tuple[str, int]]:
        return [
            (OPENROUTER_CHAT_MODEL, CHAT_MAX_TOKENS),
            (OPENROUTER_FALLBACK_MODEL, FALLBACK_MAX_TOKENS),
            (OPENROUTER_FALLBACK_MODEL_2, FALLBACK_MAX_TOKENS),
            (OPENROUTER_FALLBACK_MODEL_3, FALLBACK_MAX_TOKENS),
        ]

    def _call_openrouter(self, prompt: str, context: Optional[str]) -> str:
        """Call the OpenRouter chat completions endpoint, trying fallback models in order."""
        payload = {
            "messages": self._chat_messages(prompt, context),
            "temperature": 0.3,
        }

        last_error: Exception | None = None
        tried: set[str] = set()
        for model, max_tokens in self._openrouter_candidates():
            if not model or model in tried:
                continue
            tried.add(model)
            try:
                return self._post_chat_completions(
                    OPENROUTER_BASE_URL, self.api_key, model,
                    {**payload, "max_tokens": max_tokens},
                )
            except Exception as e:
                last_error = e
                logger.warning("OpenRouter model %s failed (%s); trying next", model, e)

        raise last_error or RuntimeError("No OpenRouter model available")

    def _call_groq(self, prompt: str, context: Optional[str]) -> str:
        """Call Groq's OpenAI-compatible chat completions endpoint."""
        payload = {
            "messages": self._chat_messages(prompt, context),
            "temperature": 0.3,
            "max_tokens": CHAT_MAX_TOKENS,
        }
        return self._post_chat_completions(GROQ_BASE_URL, GROQ_API_KEY, GROQ_MODEL, payload)

    def _post_chat_completions(
        self, base_url: str, api_key: str, model: str, payload: dict[str, Any]
    ) -> str:
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, headers=headers, json={**payload, "model": model}, timeout=90)
        if resp.status_code != 200:
            raise RuntimeError(f"Chat API {model} returned {resp.status_code}: {resp.text[:200]}")
        choices = resp.json().get("choices") or []
        content = (choices[0].get("message", {}) or {}).get("content") if choices else None
        if not content:
            raise RuntimeError(f"Chat API {model} returned an empty message")
        return content

    def _stream_chat_completions(
        self, base_url: str, api_key: str, model: str, payload: dict[str, Any]
    ) -> Generator[str, None, None]:
        """Stream token deltas from an OpenAI-compatible endpoint."""
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with requests.post(
            url,
            headers=headers,
            json={**payload, "model": model, "stream": True},
            timeout=180,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Chat API {model} returned {resp.status_code}: {resp.text[:200]}"
                )
            emitted = False
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    emitted = True
                    yield content
            if not emitted:
                raise RuntimeError(f"Chat API {model} returned an empty stream")

    def _stream_openrouter(
        self, prompt: str, context: Optional[str]
    ) -> Generator[str, None, None]:
        base_payload = {
            "messages": self._chat_messages(prompt, context),
            "temperature": 0.3,
        }
        last_error: Exception | None = None
        tried: set[str] = set()
        for model, max_tokens in self._openrouter_candidates():
            if not model or model in tried:
                continue
            tried.add(model)
            emitted = False
            try:
                for chunk in self._stream_chat_completions(
                    OPENROUTER_BASE_URL, self.api_key, model,
                    {**base_payload, "max_tokens": max_tokens},
                ):
                    emitted = True
                    yield chunk
                return
            except Exception as e:
                last_error = e
                logger.warning(
                    "OpenRouter stream model %s failed (%s); trying next", model, e
                )
                if emitted:
                    raise
        raise last_error or RuntimeError("No OpenRouter model available")

    def _stream_groq(
        self, prompt: str, context: Optional[str]
    ) -> Generator[str, None, None]:
        payload = {
            "messages": self._chat_messages(prompt, context),
            "temperature": 0.3,
            "max_tokens": CHAT_MAX_TOKENS,
        }
        yield from self._stream_chat_completions(
            GROQ_BASE_URL, GROQ_API_KEY, GROQ_MODEL, payload
        )

    def _stream_google(
        self, prompt: str, context: Optional[str]
    ) -> Generator[str, None, None]:
        """Stream from the Gemini SDK, falling back to the REST endpoint."""
        full_prompt = self._build_prompt(prompt, context)
        if HAS_GENAI and self._model is not None:
            produced = False
            try:
                for chunk in self._stream_response(full_prompt):
                    produced = True
                    yield chunk
                return
            except Exception as e:
                logger.warning("Gemini SDK stream failed (%s); trying REST", e)
                if produced:
                    raise
        if self.api_key and HAS_REQUESTS:
            text = self._call_rest_api(full_prompt)
            if text:
                yield text
                return
        raise RuntimeError("Gemini returned an empty stream")

    def _stream_with_fallbacks(
        self, prompt: str, context: Optional[str], order: list[str]
    ) -> Generator[str, None, None]:
        """Stream from each configured provider in order, then fall back offline."""
        for name in order:
            if name == "openrouter" and not is_openrouter_key(self.api_key):
                continue
            if name == "groq" and not self._groq_enabled:
                continue
            if name == "google" and (self._provider != "google" or not self.api_key):
                continue
            runner = {
                "openrouter": self._stream_openrouter,
                "google": self._stream_google,
                "groq": self._stream_groq,
            }[name]
            produced = False
            try:
                for chunk in runner(prompt, context):
                    produced = True
                    yield chunk
                return
            except Exception as e:
                logger.error("%s stream error: %s", name, e)
                if produced:
                    return
        yield self._generate_offline_grounded_response(prompt, context)

    def _build_user_message(self, prompt: str, context: Optional[str]) -> str:
        parts: list[str] = []
        if context:
            parts.extend([
                "### RETRIEVED NASA RESEARCH DATA & CONTEXT:\n",
                context,
                "\n---\n",
            ])
        parts.extend(["### USER QUERY:\n", prompt, "\n\n### SCIENTIFIC ANSWER:\n"])
        return "".join(parts)

    def _stream_response(self, prompt: str) -> Generator[str, None, None]:
        """Stream chunks from the Gemini SDK model."""
        response = self._model.generate_content(prompt, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _call_rest_api(self, prompt: str) -> str:
        """Call Gemini REST API endpoint directly."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": CHAT_MAX_TOKENS}
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
        raise RuntimeError(f"REST API call failed with status {resp.status_code}: {resp.text}")

    def _generate_offline_grounded_response(self, prompt: str, context: Optional[str]) -> str:
        """Provide a clean, readable response grounded in the retrieved NASA context."""
        intro = "FlameAI Scientific Analysis (NASA Microgravity Database):\n\n"
        if not context or not context.strip():
            return (
                f"{intro}Based on microgravity combustion fundamentals, buoyancy-driven natural convection "
                f"is absent in freefall. Flames become spherical, soot agglomerate sizes increase, and "
                f"flammability boundaries shift significantly compared to Earth (1g).\n\n"
                f"Query: \"{prompt}\"\n\n"
                f"Recommendations for space habitats:\n"
                f"1. Ventilation Shutdown: In microgravity, airflow is the main driver of flame spread. Ceasing ventilation starves flames.\n"
                f"2. Low-Pressure Oxygen Risk: Exploration atmospheres (34% O2 at 56 kPa) increase material ignition sensitivity.\n"
                f"3. Detection Systems: Photoelectric detectors are required due to larger soot aggregate chains in 0g.\n\n"
                f"[Mode: Local NASA Knowledge Base]"
            )

        # Extract meaningful scientific excerpts (ignoring markdown headings and metadata lines)
        lines = [line.strip() for line in context.split("\n")]
        valid_excerpts = []
        for line in lines:
            if not line:
                continue
            # Skip markdown headers and debug tags
            if line.startswith("#") or line.startswith("=") or line.startswith("---") or line.startswith("//"):
                continue
            if line.lower().startswith("entity:") or line.lower().startswith("source") or line.lower().startswith("category:"):
                continue
            # Clean any leftover formatting
            cleaned = line.lstrip("-•* ").replace("**", "").replace("*", "")
            if len(cleaned) > 20 and cleaned not in valid_excerpts:
                valid_excerpts.append(cleaned)
            if len(valid_excerpts) >= 4:
                break

        if not valid_excerpts:
            valid_excerpts = [
                "Microgravity eliminates buoyant upward plume formation, causing flames to adopt spherical symmetry.",
                "Flame spread rates over solid materials drop by 30% to 60% compared to 1g upward spread.",
                "Radiative heat losses govern extinction in low-speed forced flows.",
            ]

        summary_bullets = "\n".join(f"• {exc}" for exc in valid_excerpts)

        return (
            f"{intro}"
            f"Regarding your query \"{prompt}\", the NASA combustion research database indicates:\n\n"
            f"Key Findings from Ingested Experiments:\n"
            f"{summary_bullets}\n\n"
            f"Fire Safety Interpretation for Space Habitats:\n"
            f"1. Convection Loss: Without buoyant natural convection, flames spread more slowly but accumulate toxic pyrolysis gases.\n"
            f"2. Extinction Thresholds: Radiative extinction occurs at higher ambient oxygen than on Earth.\n"
            f"3. Ventilation Control: Cabin ventilation dictates flame spread direction; emergency fan shutdown is the primary containment step.\n\n"
            f"[Mode: Local NASA Knowledge Base]"
        )

    def _build_prompt(self, prompt: str, context: Optional[str]) -> str:
        """Compose the prompt with system instructions and retrieved context."""
        parts = [SYSTEM_PROMPT, "\n---\n"]
        if context:
            parts.extend([
                "### RETRIEVED NASA RESEARCH DATA & CONTEXT:\n",
                context,
                "\n---\n",
            ])
        parts.extend([
            "### USER QUERY:\n",
            prompt,
            "\n\n### SCIENTIFIC ANSWER:\n",
        ])
        return "".join(parts)
