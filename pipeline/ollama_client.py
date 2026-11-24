"""Thin wrapper around the Ollama /api/generate endpoint."""

import requests
import json


OLLAMA_BASE_URL = "http://localhost:11435"


def generate(
    model: str,
    prompt: str,
    temperature: float = 0.0,
    system: str | None = None,
    timeout: int = 120,
) -> str:
    """Call Ollama and return the response text.

    Raises requests.HTTPError on non-2xx responses.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def list_models() -> list[str]:
    """Return locally available model names."""
    resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
