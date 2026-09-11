"""Print generation results with ANSI colors in a terminal or notebook.

Usage (from the repository root)::

    from utils.visualize_repairs import print_repair_result

    result = generator.generate()
    print_repair_result(result, tokenizer)

Green marks regenerated spans in ``fixed``. Each span is followed by its old
text in red: ``new text [gốc: 'old text']``. Red also marks the corresponding
old tokens found in ``unfixed``. Use the same tokenizer as generation. No extra
packages are required. Pass ``show_original=False`` to hide inline old text.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Mapping, Sequence, TextIO


_GREEN = "\033[92m"
_RED = "\033[91m"
_RESET = "\033[0m"


def _event_tokens(event: Mapping[str, Any], side: str) -> list[int]:
    """Support generate.py/generate_oop.py and self-coding.py event formats."""
    if side in event:
        token_ids = event[side]["token_ids"]
    else:
        key = "removed_token_ids" if side == "before" else "replacement_token_ids"
        token_ids = event[key]
    return [int(token_id) for token_id in token_ids]


def _repair_masks(
    fixed_ids: list[int],
    unfixed_ids: list[int] | None,
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[bool], list[bool], int]:
    fixed_mask = [False] * len(fixed_ids)
    unfixed_mask = [False] * len(unfixed_ids or [])

    for event in events:
        start = int(event["start_index"])
        if start < 0:
            raise ValueError("repair_events start_index must be non-negative.")
        # A later backtrack discards every previous highlight after its start.
        start = min(start, len(fixed_ids))
        end = min(start + len(_event_tokens(event, "after")), len(fixed_ids))
        fixed_mask[start:] = (
            [True] * (end - start) + [False] * (len(fixed_ids) - end)
        )

    if unfixed_ids is None:
        return fixed_mask, unfixed_mask, 0

    unmatched = 0
    history_ids = list(fixed_ids)
    for event in reversed(events):
        start = int(event["start_index"])
        before_ids = _event_tokens(event, "before")
        # Undo repairs in reverse order to recover the prefix at each event.
        # The rejected candidate belongs to `before`, even though it was never
        # accepted. Later tokens are irrelevant when reconstructing this prefix.
        history_ids = history_ids[:start] + before_ids
        end = start + len(before_ids)
        if not before_ids:
            continue

        # UNFIXED is an independent baseline: event indices belong to FIXED
        # at repair time. Align with the known prefix, never reuse indices
        # blindly or color the entire divergent baseline continuation.
        matcher = SequenceMatcher(a=history_ids, b=unfixed_ids, autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.a <= start and end <= block.a + block.size:
                baseline_start = block.b + start - block.a
                baseline_end = baseline_start + len(before_ids)
                unfixed_mask[baseline_start:baseline_end] = [True] * len(before_ids)
                break
        else:
            unmatched += 1

    return fixed_mask, unfixed_mask, unmatched


def _token_spans(mask: list[bool]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for index, highlighted in enumerate([*mask, False]):
        if highlighted and start is None:
            start = index
        elif not highlighted and start is not None:
            spans.append((start, index))
            start = None
    return spans


def _common_length(left: str, right: str, *, from_end: bool = False) -> int:
    pairs = zip(reversed(left), reversed(right)) if from_end else zip(left, right)
    count = 0
    for first, second in pairs:
        if first != second:
            break
        count += 1
    return count


def _visible_repairs(
    events: Sequence[Mapping[str, Any]], token_count: int, tokenizer: Any
) -> list[tuple[int, int, str]]:
    """Keep separate annotations for repairs that survive later backtracks."""
    spans = []
    limit = token_count
    for event in reversed(events):
        start = int(event["start_index"])
        after_end = start + len(_event_tokens(event, "after"))
        end = min(after_end, limit)
        if start < end:
            original = tokenizer.decode(
                _event_tokens(event, "before"), skip_special_tokens=True
            )
            # Token counts can differ, so pair whole event spans rather than
            # implying a one-to-one mapping between old and replacement tokens.
            label = "gốc" if end == after_end else "gốc của cả lần sửa"
            spans.append((start, end, f"[{label}: {original!r}]"))
        limit = min(limit, start)
    return list(reversed(spans))


def _color_text(
    branch: Mapping[str, Any],
    mask: list[bool],
    tokenizer: Any,
    color: str,
    *,
    events: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    text = branch["text"]
    token_ids = [int(token_id) for token_id in branch["token_ids"]]
    if tokenizer.decode(token_ids, skip_special_tokens=True) != text:
        raise ValueError(
            "branch['text'] does not match decoded branch['token_ids']. "
            "Use the same tokenizer and unmodified result from generate()."
        )

    token_spans = (
        _visible_repairs(events, len(token_ids), tokenizer)
        if events is not None
        else [(start, end, "") for start, end in _token_spans(mask)]
    )
    spans: list[tuple[int, int, list[str]]] = []
    for start, end, annotation in token_spans:
        if not tokenizer.decode(token_ids[start:end], skip_special_tokens=True):
            continue  # EOS and other invisible special tokens have no text.
        prefix = tokenizer.decode(token_ids[:start], skip_special_tokens=True)
        suffix = tokenizer.decode(token_ids[end:], skip_special_tokens=True)
        # Decode in context instead of concatenating individual token strings:
        # byte-level tokens can split a Unicode character. Common boundaries
        # expand the highlight to that whole character while preserving text.
        char_start = _common_length(prefix, text)
        char_end = len(text) - _common_length(suffix, text, from_end=True)
        if char_start < char_end:
            spans.append((char_start, char_end, [annotation] if annotation else []))

    merged: list[tuple[int, int, list[str]]] = []
    for start, end, annotations in sorted(spans, key=lambda span: span[:2]):
        # Adjacent events must keep their own old text. Merge only overlapping
        # character ranges, e.g. two events touching the same Unicode character.
        if merged and start < merged[-1][1]:
            previous_start, previous_end, previous_annotations = merged[-1]
            merged[-1] = (
                previous_start, max(previous_end, end),
                previous_annotations + annotations,
            )
        else:
            merged.append((start, end, annotations))

    parts = []
    cursor = 0
    reset = _RESET if color else ""
    original_color = _RED if color else ""
    for start, end, annotations in merged:
        parts.extend((text[cursor:start], color, text[start:end], reset))
        for annotation in annotations:
            parts.extend((" ", original_color, annotation, reset))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def format_repair_result(
    result: Mapping[str, Any],
    tokenizer: Any,
    *,
    color: bool = True,
    show_original: bool = True,
) -> str:
    """Return both texts with repair spans highlighted using ANSI escapes.

    Supports both ``before/after.token_ids`` and
    ``removed_token_ids/replacement_token_ids`` events. Green covers the
    accepted regeneration span recorded by each event, including any safe
    closing tokens. Later backtracks remove obsolete green highlights.

    By default, each visible fixed span is followed by ``[gốc: 'old text']``
    in red, using that event's ``before`` tokens, even if they cannot be found
    in UNFIXED. Adjacent events are annotated separately. If a later repair
    removes part of a span, the annotation explicitly refers to the entire
    earlier event. Pass ``show_original=False`` to hide these annotations.

    Red is a best-effort token alignment: a complete ``before`` span must
    match the independent baseline within a matching block. Repeated tokens
    can make alignment ambiguous; absent spans are reported below the texts.
    Differences outside repair events are not highlighted.

    Branch text, whitespace and Unicode are preserved around inserted notes.
    Old text is quoted, with newlines/tabs escaped for inline readability.
    Invisible special tokens are skipped. ``unfixed=None`` is supported.
    Set ``color=False`` for plain text output, including inline annotations.
    """
    fixed = result["fixed"]
    unfixed = result.get("unfixed")
    events = result.get("repair_events", [])
    fixed_mask, unfixed_mask, unmatched = _repair_masks(
        [int(token_id) for token_id in fixed["token_ids"]],
        [int(token_id) for token_id in unfixed["token_ids"]]
        if unfixed is not None else None,
        events,
    )

    fixed_text = (
        _color_text(
            fixed, fixed_mask, tokenizer, _GREEN if color else "",
            events=events if show_original else None,
        )
        if color or show_original else fixed["text"]
    )
    fixed_header = "FIXED (green = regenerated tokens)"
    if show_original:
        fixed_header = "FIXED (green = regenerated tokens; red [gốc] = original text)"
    sections = [fixed_header, fixed_text]
    if unfixed is None:
        sections.extend(("", "UNFIXED: unavailable (include_unfixed=False)."))
    else:
        unfixed_text = (
            _color_text(unfixed, unfixed_mask, tokenizer, _RED)
            if color else unfixed["text"]
        )
        sections.extend((
            "", "UNFIXED (red = original tokens matched to repair events)", unfixed_text,
        ))
        if unmatched:
            sections.extend((
                "",
                f"Note: {unmatched}/{len(events)} repair event(s) could not be matched "
                "to UNFIXED; their original spans are not highlighted.",
            ))
    return "\n".join(sections)


def print_repair_result(
    result: Mapping[str, Any],
    tokenizer: Any,
    *,
    color: bool = True,
    show_original: bool = True,
    file: TextIO | None = None,
) -> None:
    """Print ``format_repair_result(...)`` to stdout or a supplied text stream."""
    print(
        format_repair_result(
            result, tokenizer, color=color, show_original=show_original
        ),
        file=file,
    )
