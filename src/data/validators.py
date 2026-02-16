"""Pydantic models for validating Tausi API responses and case data.

These models enforce data quality at ingestion time. Cases that fail
validation are logged and excluded rather than silently corrupting
the dataset.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class OutcomeLabel(str, Enum):
    """Outcome categories derived from Kenyan judgment text."""

    APPLICATION_ALLOWED = "APPLICATION_ALLOWED"
    APPLICATION_DISMISSED = "APPLICATION_DISMISSED"
    JUDGMENT_FOR_PLAINTIFF = "JUDGMENT_FOR_PLAINTIFF"
    JUDGMENT_FOR_DEFENDANT = "JUDGMENT_FOR_DEFENDANT"
    SUIT_DISMISSED = "SUIT_DISMISSED"
    STRUCK_OUT = "STRUCK_OUT"
    APPEAL_ALLOWED = "APPEAL_ALLOWED"
    APPEAL_DISMISSED = "APPEAL_DISMISSED"
    CONSENT_ORDER = "CONSENT_ORDER"
    WITHDRAWN = "WITHDRAWN"
    PARTIAL = "PARTIAL"
    UNDETERMINED = "UNDETERMINED"


# Binary mapping
POSITIVE_OUTCOMES = {
    OutcomeLabel.APPLICATION_ALLOWED,
    OutcomeLabel.JUDGMENT_FOR_PLAINTIFF,
    OutcomeLabel.APPEAL_ALLOWED,
}
NEGATIVE_OUTCOMES = {
    OutcomeLabel.APPLICATION_DISMISSED,
    OutcomeLabel.JUDGMENT_FOR_DEFENDANT,
    OutcomeLabel.SUIT_DISMISSED,
    OutcomeLabel.STRUCK_OUT,
    OutcomeLabel.APPEAL_DISMISSED,
}
EXCLUDED_OUTCOMES = {
    OutcomeLabel.CONSENT_ORDER,
    OutcomeLabel.WITHDRAWN,
    OutcomeLabel.PARTIAL,
    OutcomeLabel.UNDETERMINED,
}


class JudgeInfo(BaseModel):
    """Judge metadata from Tausi API."""

    short_name: str
    full_name: Optional[str] = None
    title: Optional[str] = None  # e.g., "Justice", "Lady Justice"


class TausiDecisionRaw(BaseModel):
    """Raw decision from the Tausi API response.

    This is a permissive model that captures what the API returns
    without failing on missing optional fields.
    """

    # Core identifiers
    frbr_uri: str = Field(description="FRBR URI, e.g. /akn/ke/judgment/kehc/2020/123")
    short_mnc: Optional[str] = Field(None, description="Short Medium-Neutral Citation")
    full_mnc: Optional[str] = Field(None, description="Full Medium-Neutral Citation")

    # Court and classification
    court: Optional[dict] = None  # {"code": "kehc", "name": "..."}
    court_class: Optional[dict] = None
    casetype: Optional[dict] = None  # {"name": "..."}

    # Dates
    filing_year: Optional[int] = None
    delivery_year: Optional[int] = None
    judgment_date: Optional[str] = None  # Will be parsed to date

    # People
    judges: list[dict] = Field(default_factory=list)
    advocates: list[dict] = Field(default_factory=list)

    # Citations and legal references
    cited_documents: list[dict] = Field(default_factory=list)

    # Content
    content_url: Optional[str] = None  # URL to the full text

    # Metadata
    county: Optional[dict] = None
    area_of_law: Optional[dict] = None
    flags: list[dict] = Field(default_factory=list)

    @property
    def case_id(self) -> str:
        """Generate a case ID from the FRBR URI."""
        parts = self.frbr_uri.strip("/").split("/")
        # /akn/ke/judgment/kehc/2020/123 -> KEHC_2020_123
        if len(parts) >= 6:
            court = parts[3].upper()
            year = parts[4]
            number = parts[5]
            return f"{court}_{year}_{number}"
        return self.frbr_uri.replace("/", "_").strip("_")

    @property
    def court_code(self) -> Optional[str]:
        if self.court and "code" in self.court:
            return self.court["code"]
        return None

    @property
    def judge_names(self) -> list[str]:
        return [j.get("short_name", j.get("name", "")) for j in self.judges if j]

    @property
    def advocate_names(self) -> list[str]:
        return [a.get("short_name", a.get("name", "")) for a in self.advocates if a]

    @property
    def cited_statute_titles(self) -> list[str]:
        return [c.get("title", "") for c in self.cited_documents if c]


class CaseRecord(BaseModel):
    """Validated, processed case record ready for feature engineering.

    This is the cleaned version of TausiDecisionRaw, with
    derived fields and strict validation.
    """

    case_id: str
    frbr_uri: str
    court_code: str
    filing_year: int = Field(ge=2000, le=2030)
    judgment_date: Optional[date] = None
    case_type: Optional[str] = None

    # Judges
    judge_names: list[str] = Field(default_factory=list)
    primary_judge: Optional[str] = None
    num_judges: int = 0

    # Advocates - sourced from API advocates[] field AND extracted from PDF text
    # API field has low completeness; PDF extraction supplements it significantly
    advocate_names: list[str] = Field(default_factory=list)
    plaintiff_advocates: list[str] = Field(default_factory=list)
    defendant_advocates: list[str] = Field(default_factory=list)
    num_advocates: int = 0
    has_advocate_data: bool = False

    # Citations
    cited_statutes: list[str] = Field(default_factory=list)
    num_citations: int = 0
    cites_dpa: bool = False  # Cites Data Protection Act

    # Content
    has_pdf: bool = False
    has_text: bool = False
    text_length: Optional[int] = None
    text_path: Optional[str] = None

    # Outcome (populated by outcome_parser)
    outcome_label: OutcomeLabel = OutcomeLabel.UNDETERMINED
    outcome_confidence: float = 0.0
    outcome_binary: Optional[int] = None  # 1=positive, 0=negative, None=excluded

    # Monetary
    claim_amount_kes: Optional[float] = None
    has_monetary_claim: bool = False

    @model_validator(mode="after")
    def derive_fields(self):
        self.num_judges = len(self.judge_names)
        if self.judge_names:
            self.primary_judge = self.judge_names[0]
        self.num_citations = len(self.cited_statutes)
        self.cites_dpa = any(
            "data protection" in s.lower() for s in self.cited_statutes
        )
        # Advocate fields
        self.num_advocates = len(self.advocate_names)
        self.has_advocate_data = self.num_advocates > 0
        # Binary outcome
        if self.outcome_label in POSITIVE_OUTCOMES:
            self.outcome_binary = 1
        elif self.outcome_label in NEGATIVE_OUTCOMES:
            self.outcome_binary = 0
        else:
            self.outcome_binary = None
        return self

    @field_validator("filing_year")
    @classmethod
    def validate_filing_year(cls, v):
        if v < 2000 or v > 2030:
            raise ValueError(f"Filing year {v} out of reasonable range")
        return v


class DatasetStats(BaseModel):
    """Summary statistics for a dataset, used for quality reporting."""

    total_cases: int = 0
    cases_with_text: int = 0
    cases_with_pdf: int = 0
    cases_with_judges: int = 0
    cases_with_advocates: int = 0
    cases_with_citations: int = 0
    cases_with_outcome: int = 0  # non-UNDETERMINED
    cases_binary_labeled: int = 0  # has outcome_binary != None

    outcome_distribution: dict[str, int] = Field(default_factory=dict)
    year_distribution: dict[int, int] = Field(default_factory=dict)
    court_distribution: dict[str, int] = Field(default_factory=dict)

    @property
    def completeness_report(self) -> str:
        if self.total_cases == 0:
            return "No cases loaded."
        pct = lambda n: f"{n}/{self.total_cases} ({100*n/self.total_cases:.1f}%)"
        return (
            f"Dataset: {self.total_cases} cases\n"
            f"  Text available: {pct(self.cases_with_text)}\n"
            f"  PDF available: {pct(self.cases_with_pdf)}\n"
            f"  Judges listed: {pct(self.cases_with_judges)}\n"
            f"  Advocates listed: {pct(self.cases_with_advocates)}\n"
            f"  Citations listed: {pct(self.cases_with_citations)}\n"
            f"  Outcome labeled: {pct(self.cases_with_outcome)}\n"
            f"  Binary-labeled: {pct(self.cases_binary_labeled)}\n"
        )
