"""
AI package — portfolio review and economics question generation.

openai is an optional dependency loaded lazily so that importing from this
package never fails when the library is absent or the API key is not set.
"""
from __future__ import annotations

__all__ = [
    "generate_ai_review",
    "resolve_groq_api_key",
    "generate_economics_questions",
]

_SUBMODULE_MAP: dict[str, str] = {
    "generate_ai_review":           "src.ai.ai_review",
    "resolve_groq_api_key":         "src.ai.ai_review",
    "generate_economics_questions": "src.ai.economics_questions",
}


def __getattr__(name: str):
    if name not in _SUBMODULE_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(_SUBMODULE_MAP[name])
    obj = getattr(module, name)
    globals()[name] = obj
    return obj
