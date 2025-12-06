"""Central LLM provider abstraction for SmartFolio and TradingAgents."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Sequence


class ProviderError(RuntimeError):
    """Raised when the configured LLM provider encounters a fatal error."""


@dataclass
class LLMRequest:
    prompt: str
    system_prompt: Optional[str]
    model: Optional[str]
    temperature: float
    top_p: float


class _BaseProvider:
    name: str = "base"
    default_model: str = ""

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - interface only
        raise NotImplementedError


class _OpenAIProvider(_BaseProvider):
    name = "openai"
    default_model = "gpt-4.1-mini"

    def __init__(self) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderError("OPENAI_API_KEY missing from environment")
        try:
            from openai import OpenAI  # pylint: disable=import-error
        except ImportError as exc:
            raise ProviderError("openai package is not installed") from exc
        self._client = OpenAI(api_key=key)

    def generate(self, request: LLMRequest) -> str:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        model_name = request.model or self.default_model
        temperature = request.temperature
        top_p = request.top_p

        def _invoke(temp: float, tp: float):
            return self._client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temp,
                top_p=tp,
            )

        try:
            response = _invoke(temperature, top_p)
        except Exception as exc:  # noqa: BLE001
            message = getattr(exc, "message", None) or str(exc)
            if "temperature" in message and "default (1)" in message:
                print(
                    f"[WARN] OpenAI model '{model_name}' only supports default temperature. "
                    "Retrying with temperature=1.",
                    file=sys.stderr,
                )
                response = _invoke(1, 1)
            else:
                raise ProviderError(f"OpenAI request failed: {exc}") from exc
        if not response.choices:
            raise ProviderError("OpenAI returned no choices")
        message = response.choices[0].message
        text = getattr(message, "content", "") or ""
        text = text.strip()
        if not text:
            raise ProviderError("OpenAI returned empty content")
        return text


class _GeminiProvider(_BaseProvider):
    name = "gemini"
    default_model = "gemini-2.0-flash"

    def __init__(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GOOGLE_API_KEY or GEMINI_API_KEY missing from environment")
        try:
            import google.generativeai as genai  # pylint: disable=import-error
        except ImportError as exc:
            raise ProviderError("google-generativeai package is not installed") from exc
        genai.configure(api_key=api_key)
        self._genai = genai

    def generate(self, request: LLMRequest) -> str:
        llm = self._genai.GenerativeModel(
            model_name=request.model or self.default_model,
            system_instruction=request.system_prompt,
        )
        response = llm.generate_content(
            request.prompt,
            generation_config={"temperature": request.temperature, "top_p": request.top_p},
        )
        text = getattr(response, "text", "") or ""
        text = text.strip()
        if not text:
            raise ProviderError("Gemini returned empty content")
        return text


_PROVIDER_MAP = {
    "openai": _OpenAIProvider,
    "gemini": _GeminiProvider,
}


def _select_provider(name: Optional[str]) -> _BaseProvider:
    resolved = (name or os.environ.get("LLM_PROVIDER") or "openai").lower()
    provider_cls = _PROVIDER_MAP.get(resolved)
    if not provider_cls:
        raise ProviderError(f"Unknown LLM provider '{resolved}'")
    return provider_cls()


def generate_completion(
    prompt: str,
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.35,
    top_p: float = 0.9,
    provider: Optional[str] = None,
) -> str:
    """Generate free-form text using the configured provider."""

    request = LLMRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    client = _select_provider(provider)
    return client.generate(request)


def list_supported_providers() -> Sequence[str]:
    return tuple(sorted(_PROVIDER_MAP.keys()))
