"""Binary Verdict Parser — converts free-text VLM responses to Yes/No.

The VLM (Moondream2) returns natural language answers like:
    "Yes, there is a large wooden pallet directly in front of the emergency exit."
    "No, the fire exit appears to be clear and unobstructed."

This module parses such responses into a strict boolean verdict using
a multi-strategy approach:
    1. First-word check (highest confidence)
    2. Keyword scoring (affirmative vs. negative word counts)
    3. Default to False (minimize false positives)

Example:
    >>> parser = BinaryVerdictParser()
    >>> parser.parse("Yes, there is a pallet blocking the exit.")
    True
    >>> parser.parse("No, the area appears clear.")
    False
    >>> parser.parse("The exit is unobstructed and accessible.")
    False
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


class BinaryVerdictParser:
    """Parses VLM free-text responses into a strict boolean verdict.

    Strategy (in order of precedence):
        1. Check if the response starts with a clear Yes/No indicator.
        2. Count affirmative vs. negative keywords in the full response.
        3. If ambiguous or empty, default to ``False`` to minimize
           false positive alerts.

    The parser is intentionally conservative — it prefers to miss a
    real violation (false negative) over raising a false alarm. In
    security monitoring, false alerts lead to "alert fatigue" and
    operators ignoring the system entirely.

    Attributes:
        AFFIRMATIVE: Set of words indicating a positive (violation) verdict.
        NEGATIVE: Set of words indicating a negative (clear) verdict.
        FIRST_WORD_YES: Set of first-word indicators for "yes".
        FIRST_WORD_NO: Set of first-word indicators for "no".
    """

    FIRST_WORD_YES: set[str] = {"yes", "yeah", "yep", "yup", "affirmative", "certainly", "indeed"}
    FIRST_WORD_NO: set[str] = {"no", "nope", "nah", "negative", "not"}

    AFFIRMATIVE: set[str] = {
        "yes", "correct", "true", "confirmed", "affirmative",
        "indeed", "certainly", "blocking", "violating", "present",
        "visible", "detected", "occupied", "obstructed", "blocked",
        "violation", "unsafe", "danger", "dangerous", "prohibited",
        "unauthorized", "trespassing", "intrusion",
    }

    NEGATIVE: set[str] = {
        "no", "incorrect", "false", "negative", "absent",
        "clear", "empty", "unblocked", "unobstructed", "safe",
        "compliant", "authorized", "permitted", "allowed",
        "nothing", "none", "neither", "cannot", "isn't",
        "aren't", "doesn't", "don't", "wasn't", "weren't",
        "free", "accessible",
    }

    @classmethod
    def parse(cls, raw_response: str) -> bool:
        """Parse a VLM response into a binary verdict.

        Args:
            raw_response: Raw text output from the VLM.

        Returns:
            ``True`` if the response indicates a violation (alert),
            ``False`` if the response indicates clear/normal.
        """
        if not raw_response or not raw_response.strip():
            logger.warning("Empty VLM response — defaulting to False (no alert).")
            return False

        text = raw_response.strip().lower()

        # ── Strategy 1: First-word check ─────────────────────────
        # This is the most reliable signal — VLMs almost always start
        # with "Yes" or "No" when asked a yes/no question.
        words = re.findall(r"[a-z']+", text)
        if words:
            first_word = words[0]
            if first_word in cls.FIRST_WORD_YES:
                logger.debug(
                    f"Verdict: TRUE (first word '{first_word}')"
                )
                return True
            if first_word in cls.FIRST_WORD_NO:
                logger.debug(
                    f"Verdict: FALSE (first word '{first_word}')"
                )
                return False

        # ── Strategy 2: Keyword scoring ──────────────────────────
        # Count affirmative vs. negative keywords. Handle negation
        # patterns like "is not blocking" by checking for "not" before
        # affirmative words.
        affirm_count, negate_count = cls._keyword_counts(words)

        if affirm_count > negate_count:
            logger.debug(
                f"Verdict: TRUE (keywords: {affirm_count} affirm vs {negate_count} negate)"
            )
            return True

        if negate_count > affirm_count:
            logger.debug(
                f"Verdict: FALSE (keywords: {affirm_count} affirm vs {negate_count} negate)"
            )
            return False

        # ── Strategy 3: Default to False (conservative) ──────────
        logger.debug(
            f"Verdict: FALSE (ambiguous — affirm={affirm_count}, "
            f"negate={negate_count}, defaulting to no-alert)"
        )
        return False

    _NEGATORS = frozenset({"not", "no", "isn't", "aren't", "doesn't", "don't", "wasn't"})

    @classmethod
    def _keyword_counts(cls, words: list[str]) -> tuple[int, int]:
        """Distinct affirmative and negative keywords, honouring negation.

        An affirmative word counts once if it appears at least once *without* a
        negator in front of it. This used to subtract every negated occurrence
        from a count of distinct words, so "not blocked ... not blocked" scored
        -1 and could tip an otherwise even response to False; and
        :meth:`parse_with_confidence` ignored negation altogether, calling
        "is not blocking" a confident yes.
        """
        affirmed = {
            w
            for i, w in enumerate(words)
            if w in cls.AFFIRMATIVE and not (i > 0 and words[i - 1] in cls._NEGATORS)
        }
        return len(affirmed), len(set(words) & cls.NEGATIVE)

    @classmethod
    def parse_with_confidence(cls, raw_response: str) -> tuple[bool, float, str]:
        """Parse a VLM response with a confidence estimate.

        Returns:
            Tuple of ``(verdict, confidence, strategy)`` where:
            - ``verdict``: True for violation, False for clear.
            - ``confidence``: Estimated confidence in the verdict (0.0–1.0).
            - ``strategy``: Which parsing strategy was used.
        """
        if not raw_response or not raw_response.strip():
            return False, 0.0, "empty_response"

        text = raw_response.strip().lower()
        words = re.findall(r"[a-z']+", text)

        if not words:
            return False, 0.0, "no_words"

        # Strategy 1: First word
        first_word = words[0]
        if first_word in cls.FIRST_WORD_YES:
            return True, 0.95, "first_word_yes"
        if first_word in cls.FIRST_WORD_NO:
            return False, 0.95, "first_word_no"

        # Strategy 2: Keyword scoring — the same counts parse() decides on, so
        # the two can never disagree about which way a response leans.
        affirm_count, negate_count = cls._keyword_counts(words)
        total = affirm_count + negate_count

        if total > 0:
            if affirm_count > negate_count:
                confidence = min(0.85, 0.5 + (affirm_count - negate_count) * 0.1)
                return True, confidence, "keyword_scoring"
            if negate_count > affirm_count:
                confidence = min(0.85, 0.5 + (negate_count - affirm_count) * 0.1)
                return False, confidence, "keyword_scoring"

        # Default
        return False, 0.3, "default_conservative"
