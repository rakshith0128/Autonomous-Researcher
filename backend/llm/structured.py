"""Getting valid JSON out of models that promise it and then do not.

Free-tier models wrap JSON in prose, fence it in markdown, emit trailing
commas, and occasionally hand back two objects concatenated. Provider-native
`response_format` helps where it exists but is not universally supported, so
every response goes through the same defensive path regardless:

    raw text -> extract -> repair syntax -> validate -> (on failure) ask again

The final fallback is a *repair round trip* that shows the model its own broken
output and the validation error. In practice that recovers the large majority
of failures on the first retry, which matters because a failed parse three
cycles into a run is expensive.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


class StructuredOutputError(ValueError):
    """Raised when output cannot be coerced into the target schema."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _slice_from(text: str, start: int) -> str | None:
    """Return the balanced JSON span beginning at `start`, if one closes.

    Scans with a depth counter while respecting string literals and escapes,
    which is what makes this survive prose containing stray braces -- a naive
    `text[text.find('{'):text.rfind('}')+1]` does not.
    """
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _balanced_slices(text: str, limit: int = 12) -> list[str]:
    """Every balanced JSON span in `text`, longest first.

    Trying only the *first* span is not enough: a model that writes "the set
    {a,b} is fine. Answer: {...}" hands back a perfectly balanced span that is
    not the payload. Ordering by length puts the real object ahead of
    incidental braces in prose, and `limit` keeps this linear enough on the
    occasional 100KB response.
    """
    spans: list[str] = []
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        span = _slice_from(text, i)
        if span and span not in spans:
            spans.append(span)
            if len(spans) >= limit:
                break
    return sorted(spans, key=len, reverse=True)


def extract_json(text: str) -> Any:
    """Best-effort JSON recovery from a chatty completion."""
    if not text or not text.strip():
        raise StructuredOutputError("model returned empty output", raw=text)

    stripped = _strip_fences(text)
    candidates: list[str] = [stripped]
    candidates.extend(_balanced_slices(stripped))
    if stripped != text:
        candidates.extend(_balanced_slices(text))

    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    errors: list[str] = []
    for candidate in candidates:
        for attempt in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))

    raise StructuredOutputError(
        f"could not parse JSON ({errors[0] if errors else 'no candidates'})", raw=text
    )


def parse_into(text: str, schema: type[T]) -> T:
    """Parse and validate in one step, raising StructuredOutputError on failure."""
    data = extract_json(text)

    # Models frequently wrap the payload: {"result": {...}} or {"proposals": [...]}
    # when the schema itself is the inner object. Unwrap a single-key envelope
    # if the inner value validates and the outer one does not.
    try:
        return schema.model_validate(data)
    except ValidationError as outer:
        if isinstance(data, dict) and len(data) == 1:
            inner = next(iter(data.values()))
            try:
                return schema.model_validate(inner)
            except ValidationError:
                pass
        # A bare list where the schema wraps a list field is also common.
        if isinstance(data, list):
            list_fields = [
                name
                for name, field in schema.model_fields.items()
                if getattr(field.annotation, "__origin__", None) is list
            ]
            if len(list_fields) == 1:
                try:
                    return schema.model_validate({list_fields[0]: data})
                except ValidationError:
                    pass
        raise StructuredOutputError(_format_validation_error(outer), raw=text) from outer


def _format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable error text for the repair prompt."""
    lines = []
    for err in exc.errors()[:8]:
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- field `{loc}`: {err['msg']}")
    return "\n".join(lines)


def schema_instructions(schema: type[BaseModel]) -> str:
    """The system-prompt fragment describing the required output shape.

    Sends the real JSON Schema rather than a hand-written description so the
    two can never drift apart as the models evolve.
    """
    return (
        "Respond with a single JSON object and nothing else. No prose, no "
        "markdown fences, no explanation before or after.\n\n"
        "It must validate against this JSON Schema:\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )


def repair_messages(
    original: list[dict[str, str]],
    raw_output: str,
    error: str,
    schema: type[BaseModel],
) -> list[dict[str, str]]:
    """Build the follow-up conversation that shows the model its own mistake.

    Keeping the original user turn in place matters: stripping it back to just
    the error makes models "fix" the JSON by inventing new content that no
    longer answers the question.
    """
    truncated = raw_output[:4000]
    return [
        *original,
        {"role": "assistant", "content": truncated},
        {
            "role": "user",
            "content": (
                "That output could not be parsed into the required schema.\n\n"
                f"Errors:\n{error}\n\n"
                "Return the SAME content, corrected to satisfy the schema. "
                "Output only the JSON object -- no fences, no commentary.\n\n"
                f"{schema_instructions(schema)}"
            ),
        },
    ]


def json_schema_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """`response_format` payload for providers that enforce schemas natively."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": False,
        },
    }
