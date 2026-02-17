"""Tests for KenyaLawScraper: metadata extraction, outcome parsing, resume logic."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.scraper import KenyaLawScraper, ScrapedCase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scraper():
    """Scraper with zero delay for fast tests."""
    s = KenyaLawScraper(delay=0)
    # Replace the real HTTP client with a mock
    s.client = MagicMock()
    return s


# ---------------------------------------------------------------------------
# ScrapedCase dataclass
# ---------------------------------------------------------------------------


class TestScrapedCase:
    def test_defaults(self):
        case = ScrapedCase()
        assert case.case_id == ""
        assert case.judges == []
        assert case.advocate_names == []
        assert case.text_length == 0

    def test_custom_fields(self):
        case = ScrapedCase(
            case_id="KEHC_2022_100",
            filing_year=2022,
            judges=["Justice A"],
            outcome="Dismissed",
        )
        assert case.case_id == "KEHC_2022_100"
        assert case.filing_year == 2022
        assert case.outcome == "Dismissed"
        assert len(case.judges) == 1


# ---------------------------------------------------------------------------
# Metadata extraction (_extract_dt_dd)
# ---------------------------------------------------------------------------


class TestExtractDtDd:
    def test_basic_pairs(self, scraper):
        html = """
        <dl>
            <dt>Citation</dt>
            <dd>Test v Test [2022] KEHC 100</dd>
            <dt>Outcome</dt>
            <dd>Dismissed</dd>
            <dt>Judges</dt>
            <dd>Justice Kimaru</dd>
        </dl>
        """
        result = scraper._extract_dt_dd(html)
        assert result["Citation"] == "Test v Test [2022] KEHC 100"
        assert result["Outcome"] == "Dismissed"
        assert result["Judges"] == "Justice Kimaru"

    def test_html_tags_stripped(self, scraper):
        html = """
        <dt><strong>Case number</strong></dt>
        <dd class="meta"><a href="#">Civil Case 123</a></dd>
        """
        result = scraper._extract_dt_dd(html)
        assert result["Case number"] == "Civil Case 123"

    def test_whitespace_cleaned(self, scraper):
        html = """
        <dt>Outcome</dt>
        <dd>  Dismissed\xa0with\xa0costs  </dd>
        """
        result = scraper._extract_dt_dd(html)
        assert result["Outcome"] == "Dismissed with costs"

    def test_empty_html(self, scraper):
        assert scraper._extract_dt_dd("") == {}
        assert scraper._extract_dt_dd("<div>no dt/dd</div>") == {}


# ---------------------------------------------------------------------------
# Judgment text extraction (_extract_judgment_text)
# ---------------------------------------------------------------------------


class TestExtractJudgmentText:
    def test_akn_judgment_body(self, scraper):
        html = """
        <section>
            <div class="akn-judgmentBody">
                <p>This is the <strong>judgment</strong> text.</p>
                <p>Second paragraph.</p>
            </div>
        </section>
        """
        text = scraper._extract_judgment_text(html)
        assert "This is the judgment text." in text
        assert "Second paragraph." in text
        # HTML tags should be stripped
        assert "<p>" not in text
        assert "<strong>" not in text

    def test_document_content_layout(self, scraper):
        html = """
        <div id="document_content" class="content">
            <p>Older style judgment content here.</p>
        </div>
        </div>
        <la-gutter></la-gutter>
        """
        text = scraper._extract_judgment_text(html)
        assert "Older style judgment content here." in text

    def test_empty_returns_empty(self, scraper):
        assert scraper._extract_judgment_text("") == ""
        assert scraper._extract_judgment_text("<div>no judgment body</div>") == ""


# ---------------------------------------------------------------------------
# Outcome extraction from text (_extract_outcome_from_text)
# ---------------------------------------------------------------------------


class TestExtractOutcomeFromText:
    """Tests for regex-based outcome extraction from judgment tail."""

    def _make_long_text(self, tail_content: str) -> str:
        """Build a text where tail_content appears in the last 20%."""
        padding = "This is preliminary analysis of the case. " * 50
        return padding + tail_content

    def test_application_dismissed(self, scraper):
        text = self._make_long_text("The application is hereby dismissed with costs.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_application_allowed(self, scraper):
        text = self._make_long_text("The application is allowed.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_suit_dismissed(self, scraper):
        text = self._make_long_text("The suit is hereby dismissed for want of prosecution.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_appeal_allowed(self, scraper):
        text = self._make_long_text("The appeal is hereby allowed.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_struck_out(self, scraper):
        text = self._make_long_text("The application is hereby struck out.")
        assert scraper._extract_outcome_from_text(text) == "Struck out"

    def test_struck_off(self, scraper):
        text = self._make_long_text("The application is struck off.")
        assert scraper._extract_outcome_from_text(text) == "Struck out"

    def test_judgment_for_plaintiff(self, scraper):
        text = self._make_long_text(
            "Judgment is hereby entered for the plaintiff in the sum of KES 5,000,000."
        )
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_judgment_for_defendant(self, scraper):
        text = self._make_long_text("Judgment is entered for the defendant with costs.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_find_for_plaintiff(self, scraper):
        text = self._make_long_text("I find for the plaintiff.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_find_for_defendant(self, scraper):
        text = self._make_long_text("I find for the defendant.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_appeal_succeeds(self, scraper):
        text = self._make_long_text("The appeal succeeds.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_appeal_fails(self, scraper):
        text = self._make_long_text("The appeal fails.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_prayers_granted(self, scraper):
        text = self._make_long_text("The prayers are hereby granted.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_prayers_refused(self, scraper):
        text = self._make_long_text("The prayer is refused.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_claim_succeeds(self, scraper):
        text = self._make_long_text("The counterclaim succeeds.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_claim_dismissed(self, scraper):
        text = self._make_long_text("The claim is dismissed.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_hereby_dismissed_broad(self, scraper):
        text = self._make_long_text("It is hereby dismissed.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_conviction_upheld(self, scraper):
        text = self._make_long_text("The conviction is hereby upheld.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_conviction_set_aside(self, scraper):
        text = self._make_long_text("The conviction is hereby set aside.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_allowed_in_part(self, scraper):
        text = self._make_long_text("The appeal is allowed in part.")
        # "allowed in part" matches the specific pattern before broad "allowed"
        outcome = scraper._extract_outcome_from_text(text)
        assert outcome in ("Allowed", "Allowed in part")

    def test_consent_order(self, scraper):
        text = self._make_long_text("The matter is settled by consent of the parties.")
        assert scraper._extract_outcome_from_text(text) == "Settled by consent"

    def test_withdrawn(self, scraper):
        text = self._make_long_text("The case has been withdrawn by the applicant.")
        assert scraper._extract_outcome_from_text(text) == "Withdrawn"

    def test_dismiss_verb_form(self, scraper):
        text = self._make_long_text("I hereby dismiss the application.")
        assert scraper._extract_outcome_from_text(text) == "Dismissed"

    def test_allow_verb_form(self, scraper):
        text = self._make_long_text("I hereby grant the application.")
        assert scraper._extract_outcome_from_text(text) == "Allowed"

    def test_empty_text(self, scraper):
        assert scraper._extract_outcome_from_text("") == ""

    def test_short_text(self, scraper):
        assert scraper._extract_outcome_from_text("Short.") == ""

    def test_no_outcome_found(self, scraper):
        text = self._make_long_text("The court adjourns to the next date.")
        assert scraper._extract_outcome_from_text(text) == ""


# ---------------------------------------------------------------------------
# Link extraction (get_case_links)
# ---------------------------------------------------------------------------


class TestGetCaseLinks:
    def test_extracts_links(self, scraper):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <div>
            <a href="/akn/ke/judgment/kehc/2022/16964/eng@2022-12-30">Case 1</a>
            <a href="/akn/ke/judgment/kehc/2022/17000/eng@2022-11-15">Case 2</a>
            <a href="/some/other/link">Not a case</a>
        </div>
        """
        scraper.client.get.return_value = mock_response
        links = scraper.get_case_links("KEHC", 2022, page=1)
        assert len(links) == 2
        assert "/akn/ke/judgment/kehc/2022/16964/eng@2022-12-30" in links

    def test_404_returns_empty_list(self, scraper):
        mock_response = MagicMock()
        mock_response.status_code = 404
        scraper.client.get.return_value = mock_response
        links = scraper.get_case_links("KEHC", 2015, page=99)
        assert links == []

    def test_no_links_returns_empty(self, scraper):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<div>No cases found</div>"
        scraper.client.get.return_value = mock_response
        links = scraper.get_case_links("KEHC", 2022, page=1)
        assert links == []


# ---------------------------------------------------------------------------
# Resume support (load_existing_ids, _save_checkpoint)
# ---------------------------------------------------------------------------


class TestResumeSupport:
    def test_load_existing_ids_valid_file(self, tmp_path):
        data = [
            {"case_id": "KEHC_2022_100", "outcome": "Dismissed"},
            {"case_id": "KEHC_2022_101", "outcome": "Allowed"},
            {"case_id": "KEHC_2021_50", "outcome": ""},
        ]
        path = tmp_path / "scraped_cases.json"
        path.write_text(json.dumps(data))

        ids = KenyaLawScraper.load_existing_ids(path)
        assert ids == {"KEHC_2022_100", "KEHC_2022_101", "KEHC_2021_50"}

    def test_load_existing_ids_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        ids = KenyaLawScraper.load_existing_ids(path)
        assert ids == set()

    def test_load_existing_ids_corrupt_file(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all {{{")
        ids = KenyaLawScraper.load_existing_ids(path)
        assert ids == set()

    def test_save_checkpoint(self, tmp_path):
        cases = [
            {"case_id": "KEHC_2022_1", "outcome": "Dismissed"},
            {"case_id": "KEHC_2022_2", "outcome": "Allowed"},
        ]
        KenyaLawScraper._save_checkpoint(cases, tmp_path)
        path = tmp_path / "scrape_checkpoint.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert len(loaded) == 2
        assert loaded[0]["case_id"] == "KEHC_2022_1"


# ---------------------------------------------------------------------------
# Resume filtering in scrape_kehc_cases
# ---------------------------------------------------------------------------


class TestScrapeResume:
    def test_skips_already_scraped_ids(self, scraper):
        """Cases in scraped_ids should be filtered out, not re-downloaded."""
        # Mock listing page returns 3 links
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <a href="/akn/ke/judgment/kehc/2022/100/eng@2022-01-01">Case 100</a>
        <a href="/akn/ke/judgment/kehc/2022/101/eng@2022-02-01">Case 101</a>
        <a href="/akn/ke/judgment/kehc/2022/102/eng@2022-03-01">Case 102</a>
        """

        # Second call (page 2) returns 404
        mock_404 = MagicMock()
        mock_404.status_code = 404
        mock_404.text = ""

        # Case page response
        mock_case_page = MagicMock()
        mock_case_page.status_code = 200
        mock_case_page.text = """
        <dt>Outcome</dt><dd>Dismissed</dd>
        <dt>Judges</dt><dd>Justice Test</dd>
        <section><div class="akn-judgmentBody"><p>The suit is dismissed.</p></div></section>
        """

        scraper.client.get.side_effect = [mock_response, mock_404, mock_case_page]

        # Case 100 already scraped
        already = {"KEHC_2022_100"}
        cases = scraper.scrape_kehc_cases(
            years=[2022], max_per_year=5, max_pages=2, scraped_ids=already
        )

        # Should have scraped 2 new cases (101 and 102), not 100
        # But mock only returns 1 case page response, so we get 1 case
        # The important thing is that case 100 was filtered out
        assert all(c.get("case_id") != "KEHC_2022_100" for c in cases)
