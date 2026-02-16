"""Tests for data validation models."""

import pytest

from src.data.validators import (
    NEGATIVE_OUTCOMES,
    POSITIVE_OUTCOMES,
    CaseRecord,
    OutcomeLabel,
    TausiDecisionRaw,
)


class TestTausiDecisionRaw:
    """Tests for raw API response validation."""

    def test_minimal_valid(self):
        """Minimum required fields parse correctly."""
        raw = TausiDecisionRaw(frbr_uri="/akn/ke/judgment/kehc/2020/123")
        assert raw.case_id == "KEHC_2020_123"
        assert raw.court_code is None
        assert raw.judge_names == []

    def test_case_id_generation(self):
        """Case ID is correctly derived from FRBR URI."""
        raw = TausiDecisionRaw(frbr_uri="/akn/ke/judgment/keca/2021/45")
        assert raw.case_id == "KECA_2021_45"

    def test_with_judges(self):
        """Judge names are extracted from nested dicts."""
        raw = TausiDecisionRaw(
            frbr_uri="/akn/ke/judgment/kehc/2020/1",
            judges=[{"short_name": "J. Mwangi"}, {"short_name": "J. Ochieng"}],
        )
        assert raw.judge_names == ["J. Mwangi", "J. Ochieng"]

    def test_with_court(self):
        """Court code is extracted from nested dict."""
        raw = TausiDecisionRaw(
            frbr_uri="/akn/ke/judgment/kehc/2020/1",
            court={"code": "kehc", "name": "High Court"},
        )
        assert raw.court_code == "kehc"


class TestCaseRecord:
    """Tests for validated case records."""

    def test_derive_fields(self):
        """Derived fields are computed on creation."""
        record = CaseRecord(
            case_id="KEHC_2020_1",
            frbr_uri="/akn/ke/judgment/kehc/2020/1",
            court_code="kehc",
            filing_year=2020,
            judge_names=["J. Mwangi", "J. Ochieng"],
            cited_statutes=["Data Protection Act - Sec 25", "Evidence Act"],
            outcome_label=OutcomeLabel.JUDGMENT_FOR_PLAINTIFF,
            outcome_confidence=0.95,
        )
        assert record.num_judges == 2
        assert record.primary_judge == "J. Mwangi"
        assert record.num_citations == 2
        assert record.cites_dpa is True
        assert record.outcome_binary == 1

    def test_negative_outcome(self):
        """Negative outcomes map to binary 0."""
        record = CaseRecord(
            case_id="TEST_1",
            frbr_uri="/test",
            court_code="kehc",
            filing_year=2020,
            outcome_label=OutcomeLabel.APPLICATION_DISMISSED,
            outcome_confidence=0.9,
        )
        assert record.outcome_binary == 0

    def test_excluded_outcome(self):
        """Excluded outcomes map to binary None."""
        record = CaseRecord(
            case_id="TEST_2",
            frbr_uri="/test",
            court_code="kehc",
            filing_year=2020,
            outcome_label=OutcomeLabel.CONSENT_ORDER,
            outcome_confidence=0.5,
        )
        assert record.outcome_binary is None

    def test_dpa_detection(self):
        """DPA citation is detected case-insensitively."""
        record = CaseRecord(
            case_id="TEST_3",
            frbr_uri="/test",
            court_code="kehc",
            filing_year=2021,
            cited_statutes=["The data protection act 2019"],
        )
        assert record.cites_dpa is True

    def test_filing_year_validation(self):
        """Out-of-range filing years are rejected."""
        with pytest.raises(ValueError):
            CaseRecord(
                case_id="TEST",
                frbr_uri="/test",
                court_code="kehc",
                filing_year=1800,
            )


class TestOutcomeLabels:
    """Tests for outcome label categorization."""

    def test_positive_outcomes(self):
        assert OutcomeLabel.APPLICATION_ALLOWED in POSITIVE_OUTCOMES
        assert OutcomeLabel.JUDGMENT_FOR_PLAINTIFF in POSITIVE_OUTCOMES
        assert OutcomeLabel.APPEAL_ALLOWED in POSITIVE_OUTCOMES

    def test_negative_outcomes(self):
        assert OutcomeLabel.APPLICATION_DISMISSED in NEGATIVE_OUTCOMES
        assert OutcomeLabel.SUIT_DISMISSED in NEGATIVE_OUTCOMES
        assert OutcomeLabel.STRUCK_OUT in NEGATIVE_OUTCOMES
