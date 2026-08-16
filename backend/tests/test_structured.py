"""Tests for JSON recovery from badly-behaved model output.

Every case here is a failure mode observed from a real free-tier model, not a
hypothetical. They are cheap to test and expensive to debug at runtime, which
makes them exactly the right thing to pin down.
"""

from __future__ import annotations

import pytest

from backend.llm.structured import (
    StructuredOutputError,
    extract_json,
    parse_into,
    repair_messages,
    schema_instructions,
)
from backend.schemas import DomainProposalBatch, QuestionProposalBatch

PROPOSAL = '{"name":"%s","description":"d","why_emerging":"w","search_terms":["t"]}'


def _batch(n: int = 3) -> str:
    inner = ",".join(PROPOSAL % chr(65 + i) for i in range(n))
    return f'{{"proposals":[{inner}]}}'


class TestExtractJson:
    def test_clean_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_fence_without_language_tag(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_leading_and_trailing_prose(self):
        raw = 'Sure! Here is the result:\n{"a": 1}\nHope that helps.'
        assert extract_json(raw) == {"a": 1}

    def test_trailing_comma(self):
        assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}

    def test_braces_in_surrounding_prose(self):
        # The naive "first balanced span" approach picks up {a,b} and fails.
        raw = 'The set {a, b} is fine. Answer: {"a": 1}'
        assert extract_json(raw) == {"a": 1}

    def test_nested_braces_and_strings(self):
        raw = '{"note": "a } brace inside a string", "inner": {"x": [1, 2]}}'
        assert extract_json(raw)["inner"]["x"] == [1, 2]

    def test_escaped_quote_inside_string(self):
        raw = r'{"q": "he said \"hi\"", "n": 2}'
        assert extract_json(raw)["n"] == 2

    def test_empty_output_raises(self):
        with pytest.raises(StructuredOutputError):
            extract_json("   ")

    def test_unparseable_raises(self):
        with pytest.raises(StructuredOutputError):
            extract_json("there is no json here at all")


class TestParseInto:
    def test_exact_shape(self):
        got = parse_into(_batch(3), DomainProposalBatch)
        assert [p.name for p in got.proposals] == ["A", "B", "C"]

    def test_bare_list_is_wrapped(self):
        """Models routinely return the list itself when the schema wraps it."""
        inner = ",".join(PROPOSAL % chr(65 + i) for i in range(3))
        got = parse_into(f"[{inner}]", DomainProposalBatch)
        assert len(got.proposals) == 3

    def test_single_key_envelope_is_unwrapped(self):
        """e.g. {"result": {"proposals": [...]}}"""
        got = parse_into(f'{{"result": {_batch(3)}}}', DomainProposalBatch)
        assert len(got.proposals) == 3

    def test_validation_failure_is_reported_with_field_names(self):
        # min_length=3 on proposals: two is not enough.
        with pytest.raises(StructuredOutputError) as exc:
            parse_into(_batch(2), DomainProposalBatch)
        assert "proposals" in str(exc.value)

    def test_raw_output_is_retained_for_the_repair_prompt(self):
        with pytest.raises(StructuredOutputError) as exc:
            parse_into("not json", DomainProposalBatch)
        assert exc.value.raw == "not json"

    def test_question_batch_enforces_two_joins(self):
        """>=2 required_joins is what makes a question 'synthesis'."""
        one_join = (
            '{"proposals":[{"text":"q","rationale":"r",'
            '"required_joins":["only-one"],"expected_measurable":"m"}]}'
        )
        with pytest.raises(StructuredOutputError):
            parse_into(one_join, QuestionProposalBatch)


class TestRepairPrompt:
    def test_repair_preserves_the_original_request(self):
        original = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "the actual task"},
        ]
        repaired = repair_messages(original, "broken{", "field x missing", DomainProposalBatch)

        assert [m["role"] for m in repaired] == ["system", "user", "assistant", "user"]
        # Dropping the original task makes models "fix" the JSON by inventing
        # new content, so the task must survive into the repair round.
        assert any("the actual task" in m["content"] for m in repaired)
        assert "field x missing" in repaired[-1]["content"]
        assert "broken{" in repaired[-2]["content"]

    def test_oversized_output_is_truncated(self):
        repaired = repair_messages([], "x" * 10_000, "err", DomainProposalBatch)
        assert len(repaired[-2]["content"]) <= 4000

    def test_schema_instructions_embed_the_real_schema(self):
        text = schema_instructions(DomainProposalBatch)
        assert "proposals" in text and "why_emerging" in text
