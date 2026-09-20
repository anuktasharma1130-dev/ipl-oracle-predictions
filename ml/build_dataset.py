"""
build_dataset.py
================
Builds a clean, chronological match-level dataset for IPL Oracle Predictions.
Reads parsed Cricsheet match records, normalizes team and venue representations,
filters abandoned/no-result matches from the ML-ready dataset, formats target labels,
and exports the dataset and comprehensive data quality report to data/processed/.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Output directories and filenames
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.parse_cricsheet import (
    DEFAULT_DATASET_DIR,
    ParsedMatch,
    parse_all_matches,
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV_PATH = PROCESSED_DATA_DIR / "ipl_matches_base.csv"
OUTPUT_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.txt"

# Core columns required for the clean match-level dataset
CORE_COLUMNS: List[str] = [
    "match_id",
    "season",
    "date",
    "venue",
    "city",
    "team1",
    "team2",
    "toss_winner",
    "toss_decision",
    "winner",
    "team1_win",
    "match_status",
]

# Additional metadata columns preserved for reference
ADDITIONAL_COLUMNS: List[str] = [
    "team1_orig",
    "team2_orig",
    "toss_winner_orig",
    "winner_orig",
    "eliminator",
    "player_of_match",
]


def generate_data_quality_report(
    total_files: int,
    parsed_matches: List[ParsedMatch],
    failed_files: List[Tuple[str, str]],
    usable_matches_df: pd.DataFrame,
    excluded_matches: List[ParsedMatch],
) -> str:
    """Generate a comprehensive text report of dataset quality and statistics."""
    total_parsed = len(parsed_matches)
    matches_with_winner = sum(1 for m in parsed_matches if m.winner is not None)
    no_result_matches = [m for m in parsed_matches if m.match_status == "no_result"]
    super_over_matches = [m for m in parsed_matches if m.eliminator is not None]

    seasons = sorted(list(set(m.season for m in parsed_matches if m.season)))
    dates = sorted([m.date for m in parsed_matches if m.date])
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "N/A"

    unique_teams_norm = sorted(list(set(
        [m.team1 for m in parsed_matches] + [m.team2 for m in parsed_matches]
    )))
    unique_venues = sorted(list(set(m.venue for m in parsed_matches if m.venue)))

    # Missing values in usable dataset
    missing_by_col = usable_matches_df.isnull().sum()

    report_lines = [
        "=" * 60,
        "IPL ORACLE PREDICTIONS - DATA QUALITY REPORT",
        "=" * 60,
        "",
        "1. FILE INGESTION SUMMARY",
        f"  Total CSV files found:          {total_files}",
        f"  Successfully parsed files:      {total_parsed}",
        f"  Failed files:                   {len(failed_files)}",
        "",
        "2. MATCH OUTCOMES BREAKDOWN",
        f"  Total matches parsed:           {total_parsed}",
        f"  Matches with decisive winner:   {matches_with_winner}",
        f"  Matches with Super Over:        {len(super_over_matches)}",
        f"  No-result / Abandoned matches:  {len(no_result_matches)}",
        f"  ML-Ready usable matches:        {len(usable_matches_df)}",
        f"  Excluded matches:               {len(excluded_matches)}",
        "",
        "3. TEMPORAL COVERAGE",
        f"  Date range:                     {date_range}",
        f"  Total seasons available:        {len(seasons)}",
        f"  Seasons list:                   {', '.join(seasons)}",
        "",
        "4. TEAMS & VENUES",
        f"  Unique normalized teams ({len(unique_teams_norm)}):",
    ]

    for team in unique_teams_norm:
        report_lines.append(f"    - {team}")

    report_lines.append(f"\n  Unique venues count:            {len(unique_venues)}")
    report_lines.append("\n5. TARGET CLASS DISTRIBUTION (team1_win in usable dataset)")
    class_counts = usable_matches_df["team1_win"].value_counts().to_dict()
    for cls, count in class_counts.items():
        pct = (count / len(usable_matches_df)) * 100
        label = "Team 1 Win" if cls == 1 else "Team 2 Win"
        report_lines.append(f"  Class {cls} ({label}): {count} matches ({pct:.2f}%)")

    report_lines.append("\n6. MISSING VALUES BY IMPORTANT COLUMN (Usable Dataset)")
    for col, count in missing_by_col.items():
        report_lines.append(f"  {col:<20}: {count} missing")

    if failed_files:
        report_lines.append("\n7. PARSING ERRORS")
        for fpath, err in failed_files:
            report_lines.append(f"  {fpath}: {err}")
    else:
        report_lines.append("\n7. PARSING ERRORS: None (100% success)")

    report_lines.append("\n" + "=" * 60)
    return "\n".join(report_lines)


def build_dataset(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    output_csv: Path = OUTPUT_CSV_PATH,
    output_report: Path = OUTPUT_REPORT_PATH,
) -> Tuple[pd.DataFrame, str]:
    """
    Build the clean match-level base dataset from raw Cricsheet CSV files.

    Returns:
        Tuple of (clean_usable_matches_dataframe, data_quality_report_text)
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scanning Cricsheet CSV files from: {dataset_dir}")
    parsed_matches, failed_files = parse_all_matches(dataset_dir, parse_balls=False)
    total_files = len(parsed_matches) + len(failed_files)
    print(f"Parsed {len(parsed_matches)} matches ({len(failed_files)} failures)")

    # Sort matches chronologically: primary by date, secondary by match_id
    parsed_matches.sort(key=lambda m: (m.date, int(m.match_id) if m.match_id.isdigit() else m.match_id))

    # Separate usable matches (with decisive winner) and excluded matches (no result)
    usable_records: List[Dict[str, Any]] = []
    excluded_matches: List[ParsedMatch] = []

    for match in parsed_matches:
        if match.team1_win is not None and match.winner is not None:
            # Build match record
            record = {
                "match_id": match.match_id,
                "season": match.season,
                "date": match.date,
                "venue": match.venue,
                "city": match.city,
                "team1": match.team1,
                "team2": match.team2,
                "toss_winner": match.toss_winner,
                "toss_decision": match.toss_decision,
                "winner": match.winner,
                "team1_win": int(match.team1_win),
                "match_status": match.match_status,
                # Reference metadata
                "team1_orig": match.team1_orig,
                "team2_orig": match.team2_orig,
                "toss_winner_orig": match.toss_winner_orig,
                "winner_orig": match.winner_orig,
                "eliminator": match.eliminator,
                "player_of_match": match.player_of_match,
            }
            usable_records.append(record)
        else:
            excluded_matches.append(match)

    # Convert to DataFrame
    all_cols = CORE_COLUMNS + ADDITIONAL_COLUMNS
    usable_df = pd.DataFrame(usable_records, columns=all_cols)

    # Save to CSV
    usable_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Saved clean match dataset to: {output_csv}")

    # Generate and save Data Quality Report
    report_text = generate_data_quality_report(
        total_files=total_files,
        parsed_matches=parsed_matches,
        failed_files=failed_files,
        usable_matches_df=usable_df,
        excluded_matches=excluded_matches,
    )
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved data quality report to: {output_report}")

    return usable_df, report_text


if __name__ == "__main__":
    usable_df, report_text = build_dataset()
    print("\n" + report_text)
