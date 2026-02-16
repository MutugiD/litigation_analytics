"""Judgment outcome extraction from Kenyan court decision text.

This is the HIGHEST-RISK component in the pipeline. Incorrect outcome
labeling corrupts the entire model. Every labeled outcome should be
validated against the taxonomy in configs/outcomes.yaml.

Approach:
1. Extract the last 20% of the judgment text (where orders/rulings appear)
2. Apply regex patterns with confidence scoring
3. Cases below confidence threshold are flagged for manual review
4. Domain expert (Kenyan lawyer) must validate >=100 labels

Usage:
    from src.features.outcome_parser import OutcomeParser
    parser = OutcomeParser()
    result = parser.parse(judgment_text)
    print(result.label, result.confidence)
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.data.validators import OutcomeLabel

logger = logging.getLogger(__name__)


@dataclass
class OutcomeResult:
    """Result of outcome extraction from a judgment."""

    label: OutcomeLabel
    confidence: float  # 0.0 to 1.0
    matched_pattern: str | None = None
    matched_text: str | None = None  # The text that triggered the match
    needs_review: bool = False
    all_matches: list[dict] = field(default_factory=list)


@dataclass
class PatternRule:
    """A regex pattern mapped to an outcome with a confidence level."""

    pattern: re.Pattern
    outcome: OutcomeLabel
    confidence: float
    raw_pattern: str


class OutcomeParser:
    """Parse judgment text to extract case outcomes.

    Uses the pattern definitions from configs/outcomes.yaml.
    Focuses on the tail section of the judgment where orders appear.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = (
                Path(__file__).resolve().parent.parent.parent / "configs" / "outcomes.yaml"
            )

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.rules: list[PatternRule] = []
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self._load_patterns(config)

        # Binary mapping from config
        self.positive = set(config["binary_mapping"]["positive"])
        self.negative = set(config["binary_mapping"]["negative"])
        self.excluded = set(config["binary_mapping"]["excluded"])

    def _load_patterns(self, config: dict):
        """Load extraction patterns from config."""
        confidence_map = {
            "high_confidence": 0.95,
            "medium_confidence": 0.80,
            "low_confidence": 0.50,
        }

        for level, conf_score in confidence_map.items():
            patterns = config.get("extraction_patterns", {}).get(level, [])
            for entry in patterns:
                try:
                    compiled = re.compile(entry["pattern"])
                    outcome = OutcomeLabel(entry["outcome"])
                    self.rules.append(
                        PatternRule(
                            pattern=compiled,
                            outcome=outcome,
                            confidence=conf_score,
                            raw_pattern=entry["pattern"],
                        )
                    )
                except (re.error, ValueError) as e:
                    logger.warning("Invalid pattern: %s -> %s", entry, e)

        logger.info("Loaded %d outcome extraction patterns", len(self.rules))

    def _get_tail_section(self, text: str, fraction: float = 0.20) -> str:
        """Extract the last `fraction` of the text (where orders typically appear).

        For Kenyan judgments, the order/ruling section is almost always
        at the very end, after the analysis and reasoning sections.
        """
        if not text:
            return ""
        start = int(len(text) * (1.0 - fraction))
        return text[start:]

    def parse(self, text: str, use_tail_only: bool = True) -> OutcomeResult:
        """Extract outcome from judgment text.

        Args:
            text: Full judgment text.
            use_tail_only: If True, only search the last 20% of text.
                          Set False to search the entire document (slower, noisier).

        Returns:
            OutcomeResult with the best-match label and confidence.
        """
        if not text or len(text.strip()) < 50:
            return OutcomeResult(
                label=OutcomeLabel.UNDETERMINED,
                confidence=0.0,
                needs_review=True,
            )

        search_text = self._get_tail_section(text) if use_tail_only else text
        all_matches = []

        for rule in self.rules:
            matches = list(rule.pattern.finditer(search_text))
            for match in matches:
                all_matches.append(
                    {
                        "label": rule.outcome,
                        "confidence": rule.confidence,
                        "pattern": rule.raw_pattern,
                        "matched_text": match.group(0),
                        "position": match.start(),
                    }
                )

        if not all_matches:
            # Try full text as fallback if tail-only was used
            if use_tail_only:
                return self.parse(text, use_tail_only=False)
            return OutcomeResult(
                label=OutcomeLabel.UNDETERMINED,
                confidence=0.0,
                needs_review=True,
                all_matches=[],
            )

        # Pick the best match: highest confidence, then latest position (closer to end)
        best = max(all_matches, key=lambda m: (m["confidence"], m["position"]))

        result = OutcomeResult(
            label=best["label"],
            confidence=best["confidence"],
            matched_pattern=best["pattern"],
            matched_text=best["matched_text"],
            needs_review=best["confidence"] < self.confidence_threshold,
            all_matches=all_matches,
        )

        # Log conflicts: if multiple different outcomes were matched
        unique_labels = set(m["label"] for m in all_matches)
        if len(unique_labels) > 1:
            logger.info(
                "Conflicting outcomes detected: %s (chose %s at %.2f)",
                unique_labels,
                result.label,
                result.confidence,
            )
            result.needs_review = True

        return result

    def to_binary(self, label: OutcomeLabel) -> int | None:
        """Map an outcome label to binary (1=positive, 0=negative, None=excluded)."""
        if label.value in self.positive:
            return 1
        elif label.value in self.negative:
            return 0
        return None

    def batch_parse(self, texts: dict[str, str]) -> dict[str, OutcomeResult]:
        """Parse outcomes for multiple cases.

        Args:
            texts: Dict mapping case_id -> judgment text

        Returns:
            Dict mapping case_id -> OutcomeResult
        """
        results = {}
        review_count = 0

        for case_id, text in texts.items():
            result = self.parse(text)
            results[case_id] = result
            if result.needs_review:
                review_count += 1

        logger.info(
            "Parsed %d cases: %d need manual review (%.1f%%)",
            len(results),
            review_count,
            100 * review_count / max(len(results), 1),
        )
        return results
