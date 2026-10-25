"""OpenAI 兼容 API 封装。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.config import require_llm_config


class LLMClient:
    def __init__(
        self,
        cfg: dict[str, Any],
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ):
        require_llm_config(cfg)
        llm = cfg["llm"]
        self.model = model or llm["model"]
        self.temperature = float(llm.get("temperature", 0.1))
        self.max_tokens = int(
            max_tokens if max_tokens is not None else llm.get("max_tokens", 1024)
        )
        self.extra_body = llm.get("extra_body") or None
        self.client = OpenAI(
            base_url=llm["base_url"],
            api_key=llm["api_key"],
            timeout=float(timeout if timeout is not None else llm.get("timeout", 60)),
            max_retries=int(
                max_retries if max_retries is not None else llm.get("max_retries", 3)
            ),
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self.client.chat.completions.create(**kwargs)
        usage = response.usage
        return {
            "content": response.choices[0].message.content or "",
            "model": response.model,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            },
        }
