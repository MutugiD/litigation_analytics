"""Web scraper for Kenya Law (new.kenyalaw.org) judgment data.

Fetches KEHC (High Court) judgments from the public website when API access
is unavailable. Extracts structured metadata + judgment text from case pages.

Data fields extracted:
    - case_id, frbr_uri, citation (MNC)
    - court_code, court_station, court_division
    - judges, case_number, case_type/nature
    - outcome (Dismissed, Allowed, etc.)
    - judgment_date, filing_year
    - judgment text (full), text_length
    - PDF download URL
    - advocate names (from judgment text header)

Usage:
    from src.data.scraper import KenyaLawScraper
    scraper = KenyaLawScraper()
    cases = scraper.scrape_kehc_cases(years=[2020, 2021, 2022], max_per_year=100)
"""

import logging
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://new.kenyalaw.org"


@dataclass
class ScrapedCase:
    """A single scraped case with all metadata."""
    case_id: str = ""
    frbr_uri: str = ""
    citation: str = ""
    mnc: str = ""
    court_code: str = ""
    court_station: str = ""
    court_division: str = ""
    case_number: str = ""
    case_type: str = ""
    case_action: str = ""
    outcome: str = ""
    judges: list[str] = field(default_factory=list)
    primary_judge: str = ""
    judgment_date: str = ""
    filing_year: int = 0
    date_published: str = ""
    filing_county: str = ""
    language: str = "English"
    order_text: str = ""  # The "Order" field from metadata
    judgment_text: str = ""
    text_length: int = 0
    pdf_url: str = ""
    page_url: str = ""
    # Advocate data extracted from judgment text
    advocate_names: list[str] = field(default_factory=list)
    plaintiff_advocates: list[str] = field(default_factory=list)
    defendant_advocates: list[str] = field(default_factory=list)


class KenyaLawScraper:
    """Scraper for new.kenyalaw.org judgment pages."""

    def __init__(self, delay: float = 1.0):
        """
        Args:
            delay: Seconds between requests (be respectful to the server).
        """
        self.delay = delay
        self.client = httpx.Client(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "LitigationAnalytics/1.0 (Research)"},
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _get(self, url: str) -> str:
        """Fetch a URL with rate limiting."""
        time.sleep(self.delay)
        r = self.client.get(url)
        r.raise_for_status()
        return r.text

    # -----------------------------------------------------------------
    # Listing pages
    # -----------------------------------------------------------------

    def get_case_links(self, court: str = "KEHC", year: int = 2022, page: int = 1) -> list[str]:
        """Get case page URLs from a listing page.

        Returns list of full case page paths like:
            /akn/ke/judgment/kehc/2022/16964/eng@2022-12-30
        """
        url = f"{BASE_URL}/judgments/{court}/{year}/?page={page}"
        html = self._get(url)

        # Extract case links with the eng@date suffix
        pattern = rf'/akn/ke/judgment/{court.lower()}/{year}/\d+/eng@[\d-]+'
        links = list(set(re.findall(pattern, html)))
        links.sort()
        logger.info("Page %d of %s/%d: found %d case links", page, court, year, len(links))
        return links

    def get_all_case_links(
        self, court: str = "KEHC", year: int = 2022, max_pages: int = 10
    ) -> list[str]:
        """Get all case links for a court/year, paginating through results."""
        all_links = []
        for page in range(1, max_pages + 1):
            links = self.get_case_links(court, year, page)
            if not links:
                break
            all_links.extend(links)
            logger.info("Year %d page %d: %d links (total: %d)", year, page, len(links), len(all_links))
        return all_links

    # -----------------------------------------------------------------
    # Case page parsing
    # -----------------------------------------------------------------

    def parse_case_page(self, path: str) -> ScrapedCase:
        """Parse a single case page and extract all metadata + text.

        Args:
            path: Case path like /akn/ke/judgment/kehc/2022/16964/eng@2022-12-30
        """
        url = f"{BASE_URL}{path}"
        html = self._get(url)
        case = ScrapedCase(page_url=url)

        # FRBR URI and case_id
        uri_match = re.search(r'/akn/ke/judgment/(\w+)/(\d{4})/(\d+)', path)
        if uri_match:
            court = uri_match.group(1)
            year = uri_match.group(2)
            number = uri_match.group(3)
            case.frbr_uri = f"/akn/ke/judgment/{court}/{year}/{number}"
            case.case_id = f"{court.upper()}_{year}_{number}"
            case.court_code = court
            case.filing_year = int(year)

        # Extract dt/dd metadata pairs
        metadata = self._extract_dt_dd(html)
        case.citation = metadata.get("Citation", "")
        case.mnc = metadata.get("Media Neutral Citation", "")
        case.court_station = metadata.get("Court station", "")
        case.court_division = metadata.get("Court division", "")
        case.case_number = metadata.get("Case number", "").replace("\xa0", " ")
        case.case_type = metadata.get("Type", "")
        case.case_action = metadata.get("Case action", "")
        case.outcome = metadata.get("Outcome", "")
        case.judgment_date = metadata.get("Judgment date", "")
        case.date_published = metadata.get("Date published", "")
        case.filing_county = metadata.get("Filing county", "")
        case.language = metadata.get("Language", "English")
        case.order_text = metadata.get("Order", "")

        # Judges - from metadata and from akn-div
        judge_text = metadata.get("Judges", "")
        if judge_text:
            case.judges = [j.strip() for j in re.split(r'[,&]', judge_text) if j.strip()]
        # Also try akn-div judges
        akn_judges = re.findall(r'class="akn-div judges">(.*?)</div>', html)
        if akn_judges:
            for j_html in akn_judges:
                j_clean = re.sub(r'<[^>]+>', '', j_html).strip()
                # Remove title suffixes like ", J" or ", JA"
                j_name = re.sub(r',\s*J[A]?$', '', j_clean).strip()
                if j_name and j_name not in case.judges:
                    case.judges.append(j_name)
        if case.judges:
            case.primary_judge = case.judges[0]

        # Judgment text
        case.judgment_text = self._extract_judgment_text(html)
        case.text_length = len(case.judgment_text)

        # PDF URL
        pdf_match = re.search(r'(https?://[^"]+\.pdf[^"]*)', html)
        if pdf_match:
            case.pdf_url = pdf_match.group(1)

        # Extract advocate names from judgment text + metadata
        if case.judgment_text:
            from src.features.lawyer_features import extract_advocates_from_text
            adv = extract_advocates_from_text(case.judgment_text)
            case.plaintiff_advocates = adv["plaintiff_advocates"]
            case.defendant_advocates = adv["defendant_advocates"]
            case.advocate_names = adv["all_advocates"]

        # Also try "Attorneys" metadata field (available on many pages)
        attorneys_text = metadata.get("Attorneys", "")
        if attorneys_text and not case.advocate_names:
            names = [n.strip() for n in re.split(r'[,;&]', attorneys_text) if n.strip()]
            case.advocate_names = names

        # If no outcome from metadata, try extracting from judgment text
        if not case.outcome and case.judgment_text:
            case.outcome = self._extract_outcome_from_text(case.judgment_text)

        return case

    def _extract_dt_dd(self, html: str) -> dict[str, str]:
        """Extract all dt/dd metadata pairs from HTML."""
        pairs = re.findall(
            r'<dt>\s*(.*?)\s*</dt>\s*<dd[^>]*>\s*(.*?)\s*</dd>',
            html, re.DOTALL,
        )
        result = {}
        for dt, dd in pairs:
            key = re.sub(r'<[^>]+>', '', dt).strip()
            val = re.sub(r'<[^>]+>', '', dd).strip()
            # Clean up whitespace and non-breaking spaces
            val = re.sub(r'\s+', ' ', val.replace('\xa0', ' ')).strip()
            if key and val:
                result[key] = val
        return result

    def _extract_judgment_text(self, html: str) -> str:
        """Extract the judgment body text from HTML.

        Handles two page layouts:
        1. Newer cases: akn-judgmentBody div (AKN markup)
        2. Older/standard cases: id="document_content" div (plain HTML)
        """
        # Try akn-judgmentBody first (newer cases with AKN markup)
        body = re.search(
            r'class="akn-judgmentBody">(.*?)</div>\s*(?:</div>\s*)*</section>',
            html, re.DOTALL,
        )
        if not body:
            body = re.search(
                r'class="akn-judgmentBody">(.*?)(?:<la-gutter|<div class="enrichments)',
                html, re.DOTALL,
            )
        if body:
            text = re.sub(r'<[^>]+>', ' ', body.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Standard layout: id="document_content" div
        body = re.search(
            r'id="document_content"[^>]*>(.*?)</div>\s*</div>\s*(?:<la-gutter|<div class="enrichments)',
            html, re.DOTALL,
        )
        if body:
            text = re.sub(r'<[^>]+>', ' ', body.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        # Broader fallback: document_content to end, trim at footer markers
        body = re.search(
            r'id="document_content"[^>]*>(.*?)$',
            html, re.DOTALL,
        )
        if body:
            text = re.sub(r'<[^>]+>', ' ', body.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            # Trim at footer markers
            for marker in ['National Council for Law Reporting', 'ISO 9001', 'Creative Commons']:
                idx = text.find(marker)
                if idx > 0:
                    text = text[:idx].strip()
                    break
            if len(text) > 100:
                return text

        # Last resort: akn-body content
        body = re.search(r'class="akn-body">(.*?)</section>', html, re.DOTALL)
        if body:
            text = re.sub(r'<[^>]+>', ' ', body.group(1))
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        return ""

    def _extract_outcome_from_text(self, text: str) -> str:
        """Extract case outcome from the last 20% of judgment text.

        Kenyan judgments typically state the outcome near the end:
        "application is dismissed", "suit is allowed", "appeal succeeds", etc.
        """
        if not text or len(text) < 200:
            return ""

        # Focus on the last 20% of text (where disposition is stated)
        tail = text[int(len(text) * 0.80):]
        tail_lower = tail.lower()

        # Ordered by specificity - check most specific patterns first
        outcome_patterns = [
            # Specific subject + outcome
            (r'(?:application|suit|appeal|petition|case|charge)\s+(?:is\s+)?(?:hereby\s+)?dismissed', "Dismissed"),
            (r'(?:application|suit|appeal|petition)\s+(?:is\s+)?(?:hereby\s+)?allowed', "Allowed"),
            (r'(?:application|suit|appeal|petition)\s+(?:is\s+)?(?:hereby\s+)?struck\s+out', "Struck out"),
            (r'(?:appeal|application)\s+(?:is\s+)?(?:hereby\s+)?(?:succeeds|successful)', "Allowed"),
            (r'(?:appeal|application)\s+(?:is\s+)?(?:hereby\s+)?(?:fails|unsuccessful)', "Dismissed"),
            (r'judgment\s+(?:is\s+)?(?:hereby\s+)?(?:entered|given)\s+for\s+the\s+plaintiff', "Allowed"),
            (r'judgment\s+(?:is\s+)?(?:hereby\s+)?(?:entered|given)\s+for\s+the\s+defendant', "Dismissed"),
            (r'(?:prayer|prayers)\s+(?:is|are)\s+(?:hereby\s+)?granted', "Allowed"),
            (r'(?:prayer|prayers)\s+(?:is|are)\s+(?:hereby\s+)?refused', "Dismissed"),
            # Verb-first patterns (judge speaking)
            (r'(?:i\s+)?(?:hereby\s+)?(?:dismiss|dismissing)\s+the\s+(?:application|suit|appeal|petition)', "Dismissed"),
            (r'(?:i\s+)?(?:hereby\s+)?(?:allow|allowing|grant|granting)\s+the\s+(?:application|suit|appeal|petition)', "Allowed"),
            # Broader patterns - "it is hereby dismissed", "is hereby dismissed"
            (r'(?:it\s+is|is)\s+hereby\s+dismissed', "Dismissed"),
            (r'(?:it\s+is|is)\s+hereby\s+allowed', "Allowed"),
            (r'hereby\s+dismissed', "Dismissed"),
            (r'hereby\s+allowed', "Allowed"),
            (r'hereby\s+upheld', "Dismissed"),  # "conviction upheld" = appeal dismissed
            (r'hereby\s+set\s+aside', "Allowed"),  # "conviction set aside" = appeal allowed
            (r'(?:conviction|sentence)\s+(?:is\s+)?(?:hereby\s+)?upheld', "Dismissed"),
            (r'(?:conviction|sentence)\s+(?:is\s+)?(?:hereby\s+)?(?:set\s+aside|quashed)', "Allowed"),
            # Partial outcomes
            (r'allowed\s+in\s+part', "Allowed in part"),
            (r'partially\s+(?:allowed|succeeded)', "Allowed in part"),
            # Consent / withdrawal
            (r'(?:by\s+consent|consent\s+order|settled)', "Settled by consent"),
            (r'withdrawn', "Withdrawn"),
            # Standalone verbs as last resort (less reliable)
            (r'\bis\s+dismissed\b', "Dismissed"),
            (r'\bis\s+allowed\b', "Allowed"),
        ]

        for pattern, outcome in outcome_patterns:
            if re.search(pattern, tail_lower):
                return outcome

        return ""

    # -----------------------------------------------------------------
    # High-level scraping
    # -----------------------------------------------------------------

    def scrape_kehc_cases(
        self,
        years: list[int] | None = None,
        max_per_year: int = 100,
        max_pages: int = 10,
    ) -> list[dict]:
        """Scrape KEHC cases for specified years.

        Args:
            years: List of years to scrape. Default: 2015-2023.
            max_per_year: Maximum cases per year.
            max_pages: Maximum listing pages per year.

        Returns:
            List of case dicts ready for DataFrame conversion.
        """
        if years is None:
            years = list(range(2015, 2024))

        all_cases = []
        for year in years:
            logger.info("Scraping KEHC %d (max %d cases)...", year, max_per_year)
            links = self.get_all_case_links("KEHC", year, max_pages=max_pages)

            # Limit to max_per_year
            links = links[:max_per_year]
            logger.info("Year %d: processing %d cases", year, len(links))

            for i, link in enumerate(links):
                try:
                    case = self.parse_case_page(link)
                    all_cases.append(asdict(case))
                    if (i + 1) % 10 == 0:
                        logger.info("  Year %d: %d/%d cases scraped", year, i + 1, len(links))
                except Exception as e:
                    logger.warning("  Failed to parse %s: %s", link, e)

        logger.info("Total scraped: %d cases across %d years", len(all_cases), len(years))
        return all_cases
