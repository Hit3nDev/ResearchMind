"""Optional Claude integration. Every call requests JSON, is parsed and then
validated with Pydantic; any failure returns None so the caller falls back to
the classical NLP implementation instead of crashing."""
import json
import logging
import re

from pydantic import BaseModel, ValidationError

from app.config import LLM_API_KEY, LLM_MODEL, llm_enabled

log = logging.getLogger("litlens.llm")
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=LLM_API_KEY, timeout=120, max_retries=3)
    return _client


def _extract_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    return json.loads(text[start:] if start > 0 else text)


def _create(**kwargs):
    import anthropic
    client = _get_client()
    try:
        # Server-side refusal fallback: if the model declines, the API reroutes to a fallback model.
        return client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                           extra_body={"fallbacks": "default"}, **kwargs)
    except anthropic.BadRequestError:
        return client.messages.create(**kwargs)


def call_text(system: str, prompt: str, effort: str = "medium") -> str | None:
    if not llm_enabled():
        return None
    import anthropic
    try:
        resp = _create(model=LLM_MODEL, max_tokens=16000, system=system,
                       output_config={"effort": effort},
                       messages=[{"role": "user", "content": prompt}])
    except anthropic.APIError as e:
        log.warning("LLM call failed: %s", e)
        return None
    if resp.stop_reason == "refusal":
        return None
    return "".join(b.text for b in resp.content if b.type == "text").strip() or None


def call_json(system: str, prompt: str, model: type[BaseModel] | None = None, effort: str = "low"):
    """Returns a validated model instance (or parsed JSON when model is None), retrying once."""
    system = system + "\nYou must respond with ONLY valid JSON matching the schema, no preamble, no markdown fences."
    for _ in range(2):
        text = call_text(system, prompt, effort)
        if text is None:
            return None
        try:
            data = _extract_json(text)
            return model.model_validate(data) if model else data
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            log.info("LLM JSON parse failed, retrying: %s", e)
    return None
