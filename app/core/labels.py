"""COCO class vocabulary and prompt inspection.

Stage 1 can only detect the eighty classes YOLO was trained on, and a rule's
``target_labels`` is what narrows the appliance to the ones an operator cares
about. Leaving that list empty means *every* class wakes the VLM — which is
occasionally what someone wants ("is anything blocking the fire exit?") and
usually a mistake, because at seconds per inference a rule about people spends
most of its budget asking about pallets and trucks.

This module spots the mistake: if the prompt names a class and no labels were
chosen, the operator almost certainly meant to scope the rule to it.
"""

from __future__ import annotations

import re

# The classes YOLOv8 is trained on. Anything outside this list cannot be a
# target label no matter what the prompt says — "forklift" and "hard hat" are
# questions for Stage 2, not filters for Stage 1.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Everyday words an operator uses for a COCO class without using its name.
# "Is this worker wearing a hard hat?" is about `person`, and the rule should
# say so rather than watching all eighty classes.
_SYNONYMS: dict[str, str] = {
    "people": "person",
    "persons": "person",
    "worker": "person",
    "workers": "person",
    "staff": "person",
    "employee": "person",
    "employees": "person",
    "man": "person",
    "men": "person",
    "woman": "person",
    "women": "person",
    "someone": "person",
    "somebody": "person",
    "anyone": "person",
    "operator": "person",
    "pedestrian": "person",
    "pedestrians": "person",
    "driver": "person",
    "vehicle": "car",
    "vehicles": "car",
    "trucks": "truck",
    "cars": "car",
    "dogs": "dog",
    "bikes": "bicycle",
    "bike": "bicycle",
    "bicycles": "bicycle",
    "boxes": "suitcase",
}

# Classes whose COCO sense is rare in surveillance prompts compared with their
# ordinary English sense. "high-visibility orange vest" is about `person`, not
# fruit; "tie" is usually a verb. Suggesting these does more harm than staying
# quiet, so they are only proposed when written as an explicit label.
_AMBIGUOUS: frozenset[str] = frozenset(
    {"orange", "tie", "bear", "train", "remote", "mouse", "clock", "apple", "kite"}
)


def labels_in_prompt(prompt: str) -> list[str]:
    """COCO classes a prompt appears to be about.

    Matching is whole-word and case-insensitive, and covers common synonyms so
    "worker" resolves to ``person``. Deliberately biased toward precision: a
    missed suggestion just leaves today's behaviour, while a wrong one sends
    the operator to narrow a rule that was already right.

    Args:
        prompt: The rule's plain-English question.

    Returns:
        Matching class names, in the order they appear in the prompt.
    """
    if not prompt:
        return []

    text = prompt.lower()
    found: list[tuple[int, str]] = []

    for term, canonical in _candidate_terms():
        if canonical in _AMBIGUOUS:
            continue
        match = re.search(rf"\b{re.escape(term)}\b", text)
        if match:
            found.append((match.start(), canonical))

    seen: set[str] = set()
    ordered: list[str] = []
    for _position, canonical in sorted(found):
        if canonical not in seen:
            seen.add(canonical)
            ordered.append(canonical)

    return ordered


def _candidate_terms() -> list[tuple[str, str]]:
    """Every searchable term paired with the class it resolves to."""
    terms = [(name, name) for name in COCO_CLASSES]
    terms.extend(_SYNONYMS.items())
    return terms


def scope_warning(prompt: str, target_labels: list[str]) -> str | None:
    """Warn when a rule watches every class but asks about a specific one.

    Returns:
        A human-readable warning, or None when the rule is already scoped or
        the prompt names nothing recognisable.
    """
    if target_labels:
        return None

    mentioned = labels_in_prompt(prompt)
    if not mentioned:
        return None

    names = ", ".join(f"'{label}'" for label in mentioned)
    return (
        f"This rule watches every object class, but its prompt is about {names}. "
        f"Every detected object — trucks, pallets, chairs — will be sent to the "
        f"VLM and judged against this question, which spends the inference "
        f"budget on things you did not ask about. Set target labels to {names} "
        f"unless you meant to watch the whole scene."
    )
