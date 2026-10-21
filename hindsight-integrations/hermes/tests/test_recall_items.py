"""Recall items — the texts behind ``recalled N memories`` (issue agentus#37).

The count alone is a claim; these are what a host expands. Two rules matter and both are cheap to
state, expensive to get wrong: the frame is capped (400 chars an item, 12K a turn) and truncation is
REPORTED, because a silently shortened memory list looks like the bank had nothing better.

There is no item COUNT cap on purpose: the total is what costs wire and SQLite bytes, and a count
cap drops whole memories even when the budget would have carried them.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from hindsight_hermes import (
    _RECALL_ITEMS_MAX_CHARS,
    _RECALL_ITEM_MAX_CHARS,
    _recall_caps,
    _recall_items,
)


def _result(text: str, **kw) -> SimpleNamespace:
    return SimpleNamespace(text=text, **kw)


def test_short_results_pass_through_untouched() -> None:
    items, truncated = _recall_items([_result("likes oolong"), _result("works on agentus")])
    assert items == [{"text": "likes oolong"}, {"text": "works on agentus"}]
    assert truncated is False


def test_tags_and_type_are_copied_not_referenced() -> None:
    """The response object is dropped after this call — copying is the whole point."""
    tags = ["preference", "tea"]
    items, _ = _recall_items([_result("oolong", tags=tags, type="preference")])
    tags.append("mutated")
    assert items == [{"text": "oolong", "tags": ["preference", "tea"], "type": "preference"}]


def test_one_long_memory_is_cut_and_says_so() -> None:
    long = "x" * (_RECALL_ITEM_MAX_CHARS + 500)
    items, truncated = _recall_items([_result(long), _result("short")])
    assert len(items[0]["text"]) == _RECALL_ITEM_MAX_CHARS + 1  # + the ellipsis
    assert items[0]["text"].endswith("…")
    assert items[1]["text"] == "short"  # the cap is per item, not a prefix-only truncation
    assert truncated is True


def test_total_cap_stops_the_list_and_says_so() -> None:
    n = (_RECALL_ITEMS_MAX_CHARS // _RECALL_ITEM_MAX_CHARS) + 5
    items, truncated = _recall_items([_result("y" * _RECALL_ITEM_MAX_CHARS)] * n)
    assert sum(len(i["text"]) for i in items) <= _RECALL_ITEMS_MAX_CHARS
    assert len(items) < n
    assert truncated is True


def test_empty_or_missing_text_is_skipped_not_counted() -> None:
    items, truncated = _recall_items([_result(""), _result("   "), SimpleNamespace(), _result("real")])
    assert items == [{"text": "real"}]
    assert truncated is False


def test_recall_status_passes_items_at_construction_and_degrades_on_an_old_host() -> None:
    """RecallStatus is FROZEN upstream, so items must arrive as constructor kwargs — assigning to
    the instance would raise on every host, ours included. A host that predates the fields raises
    TypeError there; the indicator must still show the count rather than lose the whole line.
    """

    @dataclasses.dataclass(frozen=True)
    class OldHost:
        provider_label: str = "Hindsight"
        count: int = 2
        glyph: str = "👁️"

    @dataclasses.dataclass(frozen=True)
    class NewHost:
        provider_label: str = "Hindsight"
        count: int = 2
        glyph: str = "👁️"
        items: list | None = None
        truncated: bool = False

    def build(cls, items, truncated):
        """Mirror of recall_status()'s tail: kwargs first, plain construction on TypeError."""
        if items:
            try:
                return cls(provider_label="Hindsight", count=2, glyph="👁️", items=items, truncated=truncated)
            except TypeError:
                pass
        return cls(provider_label="Hindsight", count=2, glyph="👁️")

    fresh = build(NewHost, [{"text": "a"}], True)
    assert fresh.items == [{"text": "a"}] and fresh.truncated is True
    stale = build(OldHost, [{"text": "a"}], True)
    assert stale.count == 2 and not hasattr(stale, "items")


def test_item_count_is_not_capped_only_the_total_is() -> None:
    """60 short memories fit the turn budget, so all 60 must arrive — a count cap would have
    silently dropped 40 of them while the operator's budget was never touched."""
    results = [_result(f"memory {i}") for i in range(60)]
    items, truncated = _recall_items(results)
    assert len(items) == 60
    assert truncated is False


def test_caps_default_to_the_operator_values() -> None:
    assert _RECALL_ITEM_MAX_CHARS == 400
    assert _RECALL_ITEMS_MAX_CHARS == 12_000
    assert _recall_caps({}) == (400, 12_000)


def test_caps_come_from_env_before_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host (a cockpit) overrides per spawn through the environment."""
    monkeypatch.setenv("HINDSIGHT_RECALL_ITEM_CHARS", "250")
    monkeypatch.setenv("HINDSIGHT_RECALL_TOTAL_CHARS", "9000")
    assert _recall_caps({"recallItemChars": 111, "recallTotalChars": 222}) == (250, 9000)
    monkeypatch.delenv("HINDSIGHT_RECALL_ITEM_CHARS")
    monkeypatch.delenv("HINDSIGHT_RECALL_TOTAL_CHARS")
    assert _recall_caps({"recallItemChars": 111, "recallTotalChars": 222}) == (111, 222)


def test_nonsense_caps_fall_back_to_the_default_not_to_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 / negative / junk must never mean "no cap" — one turn would become a megabyte."""
    for bad in ("0", "-5", "abc", ""):
        monkeypatch.setenv("HINDSIGHT_RECALL_ITEM_CHARS", bad)
        monkeypatch.setenv("HINDSIGHT_RECALL_TOTAL_CHARS", bad)
        assert _recall_caps({}) == (400, 12_000), bad


def test_explicit_caps_are_honoured() -> None:
    items, truncated = _recall_items([_result("z" * 50)], item_max_chars=10, total_max_chars=1000)
    assert items[0]["text"] == "z" * 10 + "…"
    assert truncated is True
    items, truncated = _recall_items([_result("a" * 30), _result("b" * 30)], total_max_chars=40)
    assert [i["text"] for i in items] == ["a" * 30]
    assert truncated is True

