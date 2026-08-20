"""Tests for COCO vocabulary and rule-scope warnings.

An empty `target_labels` means every one of the eighty COCO classes wakes the
VLM. At seconds per inference, a rule about people that watches everything
spends most of its budget judging pallets and trucks — so the appliance should
say so before the operator finds out from the alert feed.
"""

from __future__ import annotations

from app.core.labels import COCO_CLASSES, labels_in_prompt, scope_warning


class TestLabelsInPrompt:
    """Detecting what a prompt is actually about."""

    def test_a_named_class_is_found(self):
        assert labels_in_prompt("alert is person did not wear helmet") == ["person"]

    def test_synonyms_resolve_to_their_class(self):
        """Operators write "worker", not "person"."""
        assert labels_in_prompt("Is this worker wearing a hard hat?") == ["person"]
        assert labels_in_prompt("Are people blocking the aisle?") == ["person"]

    def test_matching_is_case_insensitive(self):
        assert labels_in_prompt("Is a DOG loose?") == ["dog"]

    def test_several_classes_come_back_in_prompt_order(self):
        assert labels_in_prompt("Is a person near the truck?") == ["person", "truck"]

    def test_a_class_is_only_reported_once(self):
        assert labels_in_prompt("Is the person beside another person?") == ["person"]

    def test_partial_words_do_not_match(self):
        """"personnel" is not "person", and "carpet" is not "car"."""
        assert labels_in_prompt("Check the carpet by the doorway") == []

    def test_scene_level_prompts_name_nothing(self):
        assert labels_in_prompt("Is anything blocking the fire exit?") == []
        assert labels_in_prompt("Is a pallet blocking the exit?") == []

    def test_colour_words_are_not_mistaken_for_objects(self):
        """The bundled demo asks about orange vests; that is not the fruit."""
        prompt = (
            "Is this person wearing plain dark clothing, with no "
            "high-visibility orange or yellow safety vest?"
        )
        assert labels_in_prompt(prompt) == ["person"]

    def test_empty_prompt_is_safe(self):
        assert labels_in_prompt("") == []

    def test_every_reported_label_is_a_real_coco_class(self):
        """A suggestion Stage 1 cannot act on would be worse than silence."""
        prompts = [
            "Is this worker wearing a hard hat?",
            "Is a person near the truck or the bicycle?",
            "Are people and dogs in the yard?",
        ]
        for prompt in prompts:
            for label in labels_in_prompt(prompt):
                assert label in COCO_CLASSES, f"{label} is not detectable"


class TestScopeWarning:
    """The warning fires only when the rule is genuinely misconfigured."""

    def test_unscoped_rule_about_people_is_flagged(self):
        warning = scope_warning("alert is person did not wear helmet", [])
        assert warning is not None
        assert "person" in warning

    def test_a_scoped_rule_is_not_flagged(self):
        assert scope_warning("alert is person did not wear helmet", ["person"]) is None

    def test_a_scene_level_rule_is_not_flagged(self):
        """Watching every class is correct for a question about the whole scene."""
        assert scope_warning("Is anything blocking the fire exit?", []) is None

    def test_a_rule_scoped_to_something_else_is_left_alone(self):
        """Deliberate mismatches are the operator's business, not a warning."""
        assert scope_warning("Is this person wearing a hard hat?", ["truck"]) is None
