"""Recall items — the texts behind ``recalled N memories`` (issue agentus#37).

The count alone is a claim; these are what a host expands. Two rules matter and both are cheap to
state, expensive to get wrong: the frame is capped (a 200-char item, a 6K turn) and truncation is
REPORTED, because a silently shortened memory list looks like the bank had nothing better.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

from hindsight_hermes import _RECALL_ITEMS_MAX_CHARS, _RECALL_ITEM_MAX_CHARS, _recall_items


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
