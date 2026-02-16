"""Bulk download orchestrator for Tausi API decisions.

Downloads all decisions for the configured court and year range,
saving JSON metadata and PDF judgments to data/raw/{year}/.

Supports resumable downloads - skips files that already exist on disk.

Usage:
    python -m src.data.downloader
"""

import asyncio
import json
import logging
from pathlib import Path

from tqdm import tqdm

from configs.settings import settings
from src.data.api_client import TausiClient
from src.data.validators import TausiDecisionRaw

logger = logging.getLogger(__name__)


async def download_year(
    client: TausiClient,
    year: int,
    court: str,
    output_dir: Path,
    download_pdfs: bool = True,
) -> dict:
    """Download all decisions for a given year and court.

    Returns:
        Summary dict with counts of successes, failures, and skips.
    """
    year_dir = output_dir / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    stats = {"year": year, "json_ok": 0, "json_fail": 0, "pdf_ok": 0, "pdf_fail": 0, "skipped": 0}

    async for raw_decision in client.iter_decisions(court=court, year=year):
        try:
            decision = TausiDecisionRaw.model_validate(raw_decision)
            case_id = decision.case_id

            # Save JSON metadata
            json_path = year_dir / f"{case_id}.json"
            if json_path.exists():
                stats["skipped"] += 1
                continue

            json_path.write_text(
                json.dumps(raw_decision, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            stats["json_ok"] += 1

            # Download PDF if available
            if download_pdfs and decision.frbr_uri:
                pdf_url = (
                    f"{client.base_url}/decisions{decision.frbr_uri}.pdf"
                )
                pdf_path = year_dir / f"{case_id}.pdf"
                try:
                    await client.download_file(pdf_url, pdf_path)
                    stats["pdf_ok"] += 1
                except Exception as e:
                    logger.warning("PDF download failed for %s: %s", case_id, e)
                    stats["pdf_fail"] += 1

        except Exception as e:
            logger.warning("Failed to process decision: %s", e)
            stats["json_fail"] += 1

    return stats


async def download_all(
    court: str | None = None,
    year_start: int | None = None,
    year_end: int | None = None,
    download_pdfs: bool = True,
) -> list[dict]:
    """Download all decisions across the configured year range.

    Args:
        court: Court code (default from settings)
        year_start: Start year (default from settings)
        year_end: End year (default from settings)
        download_pdfs: Whether to also download PDF judgments

    Returns:
        List of per-year stats dicts.
    """
    court = court or settings.data.target_court_code
    year_start = year_start or settings.data.filing_year_start
    year_end = year_end or settings.data.filing_year_end
    output_dir = settings.data.raw_dir

    logger.info(
        "Starting download: court=%s, years=%d-%d, pdfs=%s",
        court, year_start, year_end, download_pdfs,
    )

    all_stats = []
    async with TausiClient() as client:
        # First, get total counts for progress tracking
        total = 0
        for year in range(year_start, year_end + 1):
            count = await client.count_decisions(court=court, year=year)
            logger.info("Year %d: %d decisions available", year, count)
            total += count

        logger.info("Total decisions to process: %d", total)

        # Download year by year (sequential to respect rate limits)
        for year in tqdm(range(year_start, year_end + 1), desc="Years"):
            stats = await download_year(
                client, year, court, output_dir, download_pdfs
            )
            all_stats.append(stats)
            logger.info("Year %d complete: %s", year, stats)

    # Summary
    totals = {
        "json_ok": sum(s["json_ok"] for s in all_stats),
        "json_fail": sum(s["json_fail"] for s in all_stats),
        "pdf_ok": sum(s["pdf_ok"] for s in all_stats),
        "pdf_fail": sum(s["pdf_fail"] for s in all_stats),
        "skipped": sum(s["skipped"] for s in all_stats),
    }
    logger.info("Download complete: %s", totals)
    return all_stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(download_all())
