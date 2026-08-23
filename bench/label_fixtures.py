"""Label fixture crops by hand, so accuracy can be measured at all.

Nothing in this project has ever had ground truth. Every knob that affects a
verdict was set on a latency measurement, a geometric argument, or a single
anecdote, and the one verdict-adjacent artifact in the repo compares two models
to each other without any notion of which was right. This produces the missing
half.

Three rules about how it asks, each there to stop the labels being worse than
useless:

**It never shows the model's answer.** A labeller who sees "the VLM said Yes"
agrees with it far more often than chance. The labels would then measure
agreement with the model, which is the thing already measurable and not the
thing wanted.

**Unsure is a real answer.** A crop where a human genuinely cannot tell is not
ground truth, and coercing it to No would silently inflate precision — the
metric this project cares most about. Unsure pairs are recorded and then
excluded from the metrics.

**It asks the exact prompt Stage 2 sends**, verdict suffix included, so the
label answers the question the appliance actually asks rather than a tidied-up
paraphrase of it.

Usage:
    uv run python bench/label_fixtures.py
    uv run python bench/label_fixtures.py --rule bending
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.benchmark_vlm import PROMPTS, VERDICT_SUFFIX  # noqa: E402
from bench.make_fixtures import MANIFEST_NAME  # noqa: E402

LABELS_NAME = "labels.json"

_KEYS = {
    ord("y"): "yes",
    ord("n"): "no",
    ord("u"): "unsure",
    ord(" "): "unsure",
}
_QUIT = {27, ord("q")}
_BACK = {ord("b")}


def load_labels(path: Path) -> dict[str, str]:
    """Existing labels, keyed ``"<crop>|<rule>"``. Missing file is empty."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_labels(path: Path, labels: dict[str, str]) -> None:
    """Write after every answer, so a session can be abandoned at any point."""
    path.write_text(json.dumps(labels, indent=2, sort_keys=True))


def _canvas(crop, rule: str, prompt: str, done: int, total: int):
    """The crop with the question above it and the keys below."""
    pad_top, pad_bottom = 84, 64
    h, w = crop.shape[:2]
    canvas = cv2.copyMakeBorder(
        crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas, f"[{rule}]  {done}/{total}", (10, 26), font, 0.5, (160, 160, 160), 1
    )

    # Wrap the prompt by hand — putText has no wrapping and the PPE rule is long.
    words, line, y = prompt.split(), "", 52
    for word in words:
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, font, 0.5, 1)[0][0] > w - 20 and line:
            cv2.putText(canvas, line, (10, y), font, 0.5, (255, 255, 255), 1)
            line, y = word, y + 18
        else:
            line = trial
    if line:
        cv2.putText(canvas, line, (10, y), font, 0.5, (255, 255, 255), 1)

    cv2.putText(
        canvas,
        "y = yes   n = no   u/space = unsure",
        (10, pad_top + h + 24),
        font,
        0.5,
        (140, 200, 140),
        1,
    )
    cv2.putText(
        canvas,
        "b = back   q = save and quit",
        (10, pad_top + h + 46),
        font,
        0.5,
        (140, 140, 200),
        1,
    )
    return canvas


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", type=Path, default=Path("bench/fixtures"))
    ap.add_argument(
        "--rule",
        action="append",
        help="Limit to these rule keys. Repeat for several; default is all.",
    )
    ap.add_argument(
        "--relabel",
        action="store_true",
        help="Ask again about pairs that already have a label.",
    )
    args = ap.parse_args()

    manifest_path = args.fixtures / MANIFEST_NAME
    if not manifest_path.exists():
        raise SystemExit(
            f"No {manifest_path}. Run bench/make_fixtures.py first — and note "
            "that fixtures made by an older version were cropped differently "
            "from production, so regenerate rather than reuse."
        )
    manifest = json.loads(manifest_path.read_text())

    prompts = [(k, p) for k, p in PROMPTS if not args.rule or k in args.rule]
    if not prompts:
        raise SystemExit(f"No rules match {args.rule}. Known: {[k for k, _ in PROMPTS]}")

    labels_path = args.fixtures / LABELS_NAME
    labels = load_labels(labels_path)

    pairs = [
        (f["name"], rule, prompt)
        for f in manifest["fixtures"]
        for rule, prompt in prompts
    ]
    todo = [
        p for p in pairs if args.relabel or f"{p[0]}|{p[1]}" not in labels
    ]

    print(f"{len(pairs)} pair(s); {len(pairs) - len(todo)} already labelled.")
    if not todo:
        print("Nothing to do. Pass --relabel to revisit.")
        return
    print(f"Labelling {len(todo)}. The model's answer is deliberately not shown.")

    index = 0
    while 0 <= index < len(todo):
        name, rule, prompt = todo[index]
        crop = cv2.imread(str(args.fixtures / name))
        if crop is None:
            print(f"  missing {name}, skipping")
            index += 1
            continue

        cv2.imshow(
            "eyeq-label",
            _canvas(crop, rule, prompt + VERDICT_SUFFIX, index + 1, len(todo)),
        )
        key = cv2.waitKey(0) & 0xFF

        if key in _QUIT:
            break
        if key in _BACK:
            index = max(0, index - 1)
            continue
        if key not in _KEYS:
            continue

        labels[f"{name}|{rule}"] = _KEYS[key]
        save_labels(labels_path, labels)
        index += 1

    cv2.destroyAllWindows()
    save_labels(labels_path, labels)

    counts: dict[str, int] = {}
    for value in labels.values():
        counts[value] = counts.get(value, 0) + 1
    print(f"\nSaved {len(labels)} label(s) to {labels_path}")
    print(f"  {counts}")
    if counts.get("unsure"):
        print(
            f"  {counts['unsure']} unsure pair(s) will be excluded from the "
            "metrics rather than counted as 'no'."
        )


if __name__ == "__main__":
    main()
