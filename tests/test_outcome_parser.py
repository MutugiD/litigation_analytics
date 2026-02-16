"""Tests for outcome extraction from judgment text."""

import pytest
from src.features.outcome_parser import OutcomeParser
from src.data.validators import OutcomeLabel


@pytest.fixture
def parser():
    return OutcomeParser()


class TestOutcomeParser:
    """Tests with realistic Kenyan judgment text snippets."""

    def test_application_allowed(self, parser):
        text = """
        Having considered the submissions by both parties, I find that
        the applicant has established a prima facie case with a likelihood
        of success. The balance of convenience tilts in favour of the applicant.

        The application is hereby allowed. The respondent is restrained from
        accessing the plaintiff's personal data pending the hearing and
        determination of this suit. Costs to the applicant.
        """
        result = parser.parse(text)
        assert result.label == OutcomeLabel.APPLICATION_ALLOWED
        assert result.confidence >= 0.9
        assert not result.needs_review

    def test_application_dismissed(self, parser):
        text = """
        For the foregoing reasons, I find that the applicant has failed
        to demonstrate an arguable case or irreparable harm.

        The application is dismissed with costs to the respondent.
        """
        result = parser.parse(text)
        assert result.label == OutcomeLabel.APPLICATION_DISMISSED
        assert result.confidence >= 0.9

    def test_judgment_for_plaintiff(self, parser):
        text = """
        In conclusion, the defendant breached the terms of the contract.
        Judgment is entered for the plaintiff in the sum of KES 5,000,000
        with costs and interest at court rates.
        """
        result = parser.parse(text)
        assert result.label == OutcomeLabel.JUDGMENT_FOR_PLAINTIFF

    def test_judgment_for_defendant(self, parser):
        text = """
        The plaintiff has failed to prove its case on a balance of probabilities.
        Judgment is hereby entered for the defendant with costs.
        """
        result = parser.parse(text)
        assert result.label == OutcomeLabel.JUDGMENT_FOR_DEFENDANT

    def test_suit_dismissed(self, parser):
        text = """
        This suit has not been progressed for over two years.
        The suit is hereby dismissed for want of prosecution.
        Each party to bear its own costs.
        """
        result = parser.parse(text)
        assert result.label == OutcomeLabel.SUIT_DISMISSED

    def test_consent_order(self, parser):
        text = """
        The parties having reached an agreement, the matter is
        disposed of by consent. The terms of settlement shall
        form part of the order of this court.
        """
        result = parser.parse(text)
        assert result.label == OutcomeLabel.CONSENT_ORDER
        assert result.needs_review  # Low confidence

    def test_undetermined_short_text(self, parser):
        """Very short text returns UNDETERMINED."""
        result = parser.parse("Short text.")
        assert result.label == OutcomeLabel.UNDETERMINED
        assert result.needs_review

    def test_undetermined_empty(self, parser):
        result = parser.parse("")
        assert result.label == OutcomeLabel.UNDETERMINED

    def test_binary_mapping(self, parser):
        """Binary mapping works correctly."""
        assert parser.to_binary(OutcomeLabel.APPLICATION_ALLOWED) == 1
        assert parser.to_binary(OutcomeLabel.APPLICATION_DISMISSED) == 0
        assert parser.to_binary(OutcomeLabel.CONSENT_ORDER) is None

    def test_batch_parse(self, parser):
        """Batch parsing processes multiple texts."""
        texts = {
            "case_1": "The application is allowed with costs.",
            "case_2": "The suit is dismissed.",
            "case_3": "too short",
        }
        results = parser.batch_parse(texts)
        assert len(results) == 3
        assert results["case_1"].label == OutcomeLabel.APPLICATION_ALLOWED
        assert results["case_3"].needs_review

    def test_tail_section_focus(self, parser):
        """Parser focuses on the tail of the document where orders appear."""
        # Put a dismissal early and an allowance at the end
        text = (
            "Early in the proceedings, a preliminary application was dismissed. " * 20
            + "\n\nORDERS\n\nThe application is hereby allowed."
        )
        result = parser.parse(text)
        assert result.label == OutcomeLabel.APPLICATION_ALLOWED
