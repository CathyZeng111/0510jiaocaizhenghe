from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

DEFAULT_MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODELSCOPE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
DEFAULT_MODELSCOPE_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_OPENROUTER_CHAT_MODEL = "google/gemini-3.1-flash-lite"
_ENV_LOADED = False


class ModelScopeError(RuntimeError):
    pass


def is_modelscope_configured() -> bool:
    load_local_env()
    return bool(os.getenv("MODELSCOPE_API_KEY"))


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    response_format: Optional[dict[str, str]] = None,
) -> str:
    load_local_env()
    api_key = os.getenv("MODELSCOPE_API_KEY")
    if not api_key:
        return chat_completion_openrouter(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    base_url = os.getenv("MODELSCOPE_BASE_URL", DEFAULT_MODELSCOPE_BASE_URL).rstrip("/")
    payload: dict[str, Any] = {
        "model": model or os.getenv("MODELSCOPE_MODEL", DEFAULT_MODELSCOPE_MODEL),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        if os.getenv("OPENROUTER_API_KEY"):
            return chat_completion_openrouter(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        raise ModelScopeError(f"ModelScope HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        if os.getenv("OPENROUTER_API_KEY"):
            return chat_completion_openrouter(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        raise ModelScopeError(f"ModelScope 请求失败：{exc.reason}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        if os.getenv("OPENROUTER_API_KEY"):
            return chat_completion_openrouter(
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        raise ModelScopeError(f"ModelScope 返回结构异常：{data}") from exc


def chat_completion_openrouter(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    response_format: Optional[dict[str, str]] = None,
) -> str:
    load_local_env()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ModelScopeError("未配置 OPENROUTER_API_KEY")

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    payload: dict[str, Any] = {
        "model": model or os.getenv("OPENROUTER_CHAT_MODEL", DEFAULT_OPENROUTER_CHAT_MODEL),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ModelScopeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ModelScopeError(f"OpenRouter 请求失败：{exc.reason}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelScopeError(f"OpenRouter 返回结构异常：{data}") from exc


def create_embeddings(texts: list[str], *, model: Optional[str] = None) -> list[list[float]]:
    load_local_env()
    provider = os.getenv("EMBEDDING_PROVIDER", "modelscope").strip().lower()
    if provider == "openrouter":
        return create_openrouter_embeddings(texts, model=model)
    return create_modelscope_embeddings(texts, model=model)


def create_modelscope_embeddings(texts: list[str], *, model: Optional[str] = None) -> list[list[float]]:
    api_key = os.getenv("MODELSCOPE_API_KEY")
    if not api_key:
        raise ModelScopeError("未配置 MODELSCOPE_API_KEY")
    if not texts:
        return []

    base_url = os.getenv("MODELSCOPE_BASE_URL", DEFAULT_MODELSCOPE_BASE_URL).rstrip("/")
    payload: dict[str, Any] = {
        "model": model or os.getenv("MODELSCOPE_EMBEDDING_MODEL", DEFAULT_MODELSCOPE_EMBEDDING_MODEL),
        "input": texts,
    }

    request = urllib.request.Request(
        f"{base_url}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ModelScopeError(f"ModelScope HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ModelScopeError(f"ModelScope 请求失败：{exc.reason}") from exc

    try:
        embeddings = [
            [float(value) for value in item["embedding"]]
            for item in sorted(data["data"], key=lambda current: current.get("index", 0))
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelScopeError(f"ModelScope Embeddings 返回结构异常：{data}") from exc
    if len(embeddings) != len(texts):
        raise ModelScopeError("ModelScope Embeddings 返回数量与输入不一致")
    return embeddings


def create_openrouter_embeddings(texts: list[str], *, model: Optional[str] = None) -> list[list[float]]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ModelScopeError("未配置 OPENROUTER_API_KEY")
    if not texts:
        return []

    base_url = os.getenv("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    payload: dict[str, Any] = {
        "model": model or os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_OPENROUTER_EMBEDDING_MODEL),
        "input": texts,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    title = os.getenv("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    request = urllib.request.Request(
        f"{base_url}/embeddings",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ModelScopeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ModelScopeError(f"OpenRouter 请求失败：{exc.reason}") from exc

    try:
        embeddings = [
            [float(value) for value in item["embedding"]]
            for item in sorted(data["data"], key=lambda current: current.get("index", 0))
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelScopeError(f"OpenRouter Embeddings 返回结构异常：{data}") from exc
    if len(embeddings) != len(texts):
        raise ModelScopeError("OpenRouter Embeddings 返回数量与输入不一致")
    return embeddings


def default_embedding_model() -> str:
    load_local_env()
    if os.getenv("EMBEDDING_PROVIDER", "modelscope").strip().lower() == "openrouter":
        return os.getenv("OPENROUTER_EMBEDDING_MODEL", DEFAULT_OPENROUTER_EMBEDDING_MODEL)
    return os.getenv("MODELSCOPE_EMBEDDING_MODEL", DEFAULT_MODELSCOPE_EMBEDDING_MODEL)


def load_local_env() -> None:
    global _ENV_LOADED
    if _ENV_LOADED and os.getenv("MODELSCOPE_API_KEY"):
        return
    _ENV_LOADED = True
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ModelScopeError("模型未返回 JSON 对象")
    return json.loads(stripped[start : end + 1])
