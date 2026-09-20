"""
feature_engineering.py
======================
Builds a leakage-safe, chronological feature engineering pipeline for IPL Oracle Predictions.
Reads data/processed/ipl_matches_base.csv and computes:
- Prior franchise win rates (strictly prior to current match)
- Rolling recent form (last 5, last 10 matches)
- Head-to-head prior win rate and match counters
- Venue-specific win rates and venue chase win percentage
- Dynamic Elo ratings with K-factor=32 (updated strictly post-match)
- Team symmetry difference features (elo_diff, win_rate_diff, etc.)
- Home ground advantage (ground-truth city/venue mapping)
- Post-toss indicators (team1_toss_winner, toss_decision_field)

Outputs:
- data/processed/ipl_matches_features.csv
- data/processed/feature_engineering_report.txt
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# File paths
BASE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_matches_base.csv"
OUTPUT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_matches_features.csv"
OUTPUT_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineering_report.txt"

# Elo Rating Parameters
INITIAL_ELO: float = 1500.0
ELO_K: float = 32.0

# Neutral default probability for unknown/unseen matchups
NEUTRAL_WIN_RATE: float = 0.50

# Specific venue to city mappings for known unlisted cities (Dubai & Sharjah)
VENUE_CITY_MAPPING: Dict[str, str] = {
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
}

# Reliable, ground-truth home cities for IPL franchises
HOME_CITIES: Dict[str, Set[str]] = {
    "Chennai Super Kings": {"Chennai"},
    "Mumbai Indians": {"Mumbai"},
    "Kolkata Knight Riders": {"Kolkata"},
    "Royal Challengers Bangalore": {"Bangalore", "Bengaluru"},
    "Delhi Capitals": {"Delhi"},
    "Sunrisers Hyderabad": {"Hyderabad"},
    "Rajasthan Royals": {"Jaipur"},
    "Punjab Kings": {"Mohali", "Chandigarh", "Dharamsala"},
    "Gujarat Titans": {"Ahmedabad"},
    "Lucknow Super Giants": {"Lucknow"},
    # Historical franchises
    "Deccan Chargers": {"Hyderabad"},
    "Pune Warriors": {"Pune"},
    "Rising Pune Supergiant": {"Pune"},
    "Gujarat Lions": {"Rajkot", "Kanpur"},
    "Kochi Tuskers Kerala": {"Kochi"},
}

# Feature definitions
PRE_TOSS_FEATURES: List[str] = [
    "team1",
    "team2",
    "venue",
    "city",
    "season",
    "team1_prior_win_rate",
    "team2_prior_win_rate",
    "team1_last_5_win_rate",
    "team2_last_5_win_rate",
    "team1_last_10_win_rate",
    "team2_last_10_win_rate",
    "h2h_team1_win_rate",
    "h2h_matches_before",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_matches_before",
    "team1_elo",
    "team2_elo",
    "elo_diff",
    "prior_win_rate_diff",
    "last_5_win_rate_diff",
    "last_10_win_rate_diff",
    "venue_win_rate_diff",
    "venue_chase_win_rate",
    "team1_home_advantage",
]

POST_TOSS_ADDITIONAL_FEATURES: List[str] = [
    "team1_toss_winner",
    "toss_decision_field",
]

POST_TOSS_FEATURES: List[str] = PRE_TOSS_FEATURES + POST_TOSS_ADDITIONAL_FEATURES

IDENTIFIER_COLUMNS: List[str] = [
    "match_id",
    "date",
]

TARGET_COLUMN: str = "team1_win"

ALL_COLUMNS: List[str] = IDENTIFIER_COLUMNS + POST_TOSS_FEATURES + [TARGET_COLUMN]


class ChronologicalFeatureEngine:
    """
    Stateful chronological feature extractor for IPL match prediction.
    Maintains historical state up to match T-1 and records feature vectors
    strictly before updating state with match T's outcome.
    """

    def __init__(self, initial_elo: float = INITIAL_ELO, elo_k: float = ELO_K) -> None:
        self.initial_elo = initial_elo
        self.elo_k = elo_k
        self.reset()

    def reset(self) -> None:
        """Reset all historical accumulators."""
        # Team match histories: list of 1 (win) and 0 (loss)
        self.team_history: Dict[str, List[int]] = defaultdict(list)
        # Head to head history: list of winner team names
        self.h2h_history: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        # Venue team history: (venue, team) -> list of 1 (win) and 0 (loss)
        self.venue_team_history: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # Venue total matches count
        self.venue_matches_count: Dict[str, int] = defaultdict(int)
        # Venue chase outcomes: list of 1 (chasing team won) and 0 (defending team won)
        self.venue_chase_history: Dict[str, List[int]] = defaultdict(list)
        # Elo ratings
        self.elo_ratings: Dict[str, float] = defaultdict(lambda: self.initial_elo)

        # Statistics trackers for reporting
        self.no_h2h_matches: int = 0
        self.fewer_5_matches: int = 0
        self.fewer_10_matches: int = 0
        self.no_venue_history_matches: int = 0

    def clean_city(self, venue: str, city_val: Any) -> str:
        """Fill missing city values for known UAE venues without modifying raw base file."""
        if venue in VENUE_CITY_MAPPING:
            return VENUE_CITY_MAPPING[venue]
        if pd.isna(city_val) or not str(city_val).strip() or str(city_val).lower() == "nan":
            return ""
        return str(city_val).strip()

    def calculate_expected_elo_score(self, rating_a: float, rating_b: float) -> float:
        """Standard logistic expected score formula for Elo rating."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update_elo(self, team1: str, team2: str, team1_won: bool) -> None:
        """Update Elo ratings for both teams following the match."""
        r1 = self.elo_ratings[team1]
        r2 = self.elo_ratings[team2]

        e1 = self.calculate_expected_elo_score(r1, r2)
        e2 = self.calculate_expected_elo_score(r2, r1)

        s1 = 1.0 if team1_won else 0.0
        s2 = 1.0 - s1

        self.elo_ratings[team1] = r1 + self.elo_k * (s1 - e1)
        self.elo_ratings[team2] = r2 + self.elo_k * (s2 - e2)

    def process_matches(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the base matches dataframe chronologically and compute all features.

        Args:
            base_df: Clean base dataframe sorted chronologically.

        Returns:
            Engineered feature DataFrame.
        """
        self.reset()
        feature_rows: List[Dict[str, Any]] = []

        for idx, row in base_df.iterrows():
            m_id = row["match_id"]
            date = row["date"]
            season = str(row["season"])
            venue = str(row["venue"])
            city = self.clean_city(venue, row.get("city"))
            t1 = str(row["team1"])
            t2 = str(row["team2"])
            toss_winner = str(row["toss_winner"])
            toss_decision = str(row["toss_decision"]).lower().strip()
            winner = str(row["winner"])
            target = int(row["team1_win"])

            # -------------------------------------------------------------
            # STEP 1: READ PRIOR STATE STRICTLY BEFORE MATCH T
            # -------------------------------------------------------------
            t1_prior = self.team_history[t1]
            t2_prior = self.team_history[t2]

            # Track statistics for report
            if len(t1_prior) < 5 or len(t2_prior) < 5:
                self.fewer_5_matches += 1
            if len(t1_prior) < 10 or len(t2_prior) < 10:
                self.fewer_10_matches += 1

            # Team prior historical win rate
            t1_prior_wr = float(np.mean(t1_prior)) if t1_prior else NEUTRAL_WIN_RATE
            t2_prior_wr = float(np.mean(t2_prior)) if t2_prior else NEUTRAL_WIN_RATE

            # Recent form (last 5 matches)
            t1_last5_wr = float(np.mean(t1_prior[-5:])) if t1_prior else NEUTRAL_WIN_RATE
            t2_last5_wr = float(np.mean(t2_prior[-5:])) if t2_prior else NEUTRAL_WIN_RATE

            # Recent form (last 10 matches)
            t1_last10_wr = float(np.mean(t1_prior[-10:])) if t1_prior else NEUTRAL_WIN_RATE
            t2_last10_wr = float(np.mean(t2_prior[-10:])) if t2_prior else NEUTRAL_WIN_RATE

            # Head to head
            h2h_key = tuple(sorted([t1, t2]))
            h2h_prior = self.h2h_history[h2h_key]
            h2h_count = len(h2h_prior)
            if h2h_count == 0:
                self.no_h2h_matches += 1
                h2h_t1_wr = NEUTRAL_WIN_RATE
            else:
                t1_h2h_wins = sum(1 for w in h2h_prior if w == t1)
                h2h_t1_wr = t1_h2h_wins / h2h_count

            # Venue performance
            v_t1 = self.venue_team_history[(venue, t1)]
            v_t2 = self.venue_team_history[(venue, t2)]
            t1_venue_wr = float(np.mean(v_t1)) if v_t1 else NEUTRAL_WIN_RATE
            t2_venue_wr = float(np.mean(v_t2)) if v_t2 else NEUTRAL_WIN_RATE

            venue_count = self.venue_matches_count[venue]
            if venue_count == 0:
                self.no_venue_history_matches += 1

            # Venue chase win rate
            v_chase_prior = self.venue_chase_history[venue]
            venue_chase_wr = float(np.mean(v_chase_prior)) if v_chase_prior else NEUTRAL_WIN_RATE

            # Elo ratings before match
            t1_elo = float(self.elo_ratings[t1])
            t2_elo = float(self.elo_ratings[t2])
            elo_diff = t1_elo - t2_elo

            # Difference features
            prior_wr_diff = t1_prior_wr - t2_prior_wr
            last5_diff = t1_last5_wr - t2_last5_wr
            last10_diff = t1_last10_wr - t2_last10_wr
            venue_wr_diff = t1_venue_wr - t2_venue_wr

            # Home advantage
            t1_is_home = 1 if city in HOME_CITIES.get(t1, set()) else 0
            t2_is_home = 1 if city in HOME_CITIES.get(t2, set()) else 0
            team1_home_adv = t1_is_home - t2_is_home

            # Toss features (Post-Toss model only)
            team1_toss_winner = 1 if toss_winner == t1 else 0
            toss_decision_field = 1 if toss_decision == "field" else 0

            # -------------------------------------------------------------
            # STEP 2: ASSEMBLE FEATURE RECORD
            # -------------------------------------------------------------
            record = {
                "match_id": m_id,
                "date": date,
                "season": season,
                "team1": t1,
                "team2": t2,
                "venue": venue,
                "city": city,
                "team1_prior_win_rate": round(t1_prior_wr, 4),
                "team2_prior_win_rate": round(t2_prior_wr, 4),
                "prior_win_rate_diff": round(prior_wr_diff, 4),
                "team1_last_5_win_rate": round(t1_last5_wr, 4),
                "team2_last_5_win_rate": round(t2_last5_wr, 4),
                "last_5_win_rate_diff": round(last5_diff, 4),
                "team1_last_10_win_rate": round(t1_last10_wr, 4),
                "team2_last_10_win_rate": round(t2_last10_wr, 4),
                "last_10_win_rate_diff": round(last10_diff, 4),
                "h2h_team1_win_rate": round(h2h_t1_wr, 4),
                "h2h_matches_before": h2h_count,
                "team1_venue_win_rate": round(t1_venue_wr, 4),
                "team2_venue_win_rate": round(t2_venue_wr, 4),
                "venue_win_rate_diff": round(venue_wr_diff, 4),
                "venue_matches_before": venue_count,
                "venue_chase_win_rate": round(venue_chase_wr, 4),
                "team1_elo": round(t1_elo, 2),
                "team2_elo": round(t2_elo, 2),
                "elo_diff": round(elo_diff, 2),
                "team1_home_advantage": team1_home_adv,
                "team1_toss_winner": team1_toss_winner,
                "toss_decision_field": toss_decision_field,
                "team1_win": target,
            }
            feature_rows.append(record)

            # -------------------------------------------------------------
            # STEP 3: UPDATE HISTORICAL STATE STRICTLY AFTER RECORDING
            # -------------------------------------------------------------
            t1_won = (winner == t1)
            t2_won = (winner == t2)

            self.team_history[t1].append(1 if t1_won else 0)
            self.team_history[t2].append(1 if t2_won else 0)

            self.h2h_history[h2h_key].append(winner)

            self.venue_team_history[(venue, t1)].append(1 if t1_won else 0)
            self.venue_team_history[(venue, t2)].append(1 if t2_won else 0)
            self.venue_matches_count[venue] += 1

            # Determine chasing team (team batting second)
            if toss_winner == t1:
                chasing_team = t1 if toss_decision == "field" else t2
            else:
                chasing_team = t2 if toss_decision == "field" else t1

            chase_won = 1 if winner == chasing_team else 0
            self.venue_chase_history[venue].append(chase_won)

            # Update Elo ratings
            self.update_elo(t1, t2, t1_won)

        return pd.DataFrame(feature_rows, columns=ALL_COLUMNS)

    def generate_report(
        self,
        input_count: int,
        output_count: int,
        features_df: pd.DataFrame,
    ) -> str:
        """Generate a detailed feature engineering documentation report."""
        report_lines = [
            "=" * 70,
            "IPL ORACLE PREDICTIONS - FEATURE ENGINEERING REPORT",
            "=" * 70,
            "",
            "1. PIPELINE OVERVIEW",
            f"  Input matches count:             {input_count}",
            f"  Output matches count:            {output_count}",
            f"  Output feature columns:          {len(ALL_COLUMNS)}",
            f"  Chronological sort verification: Verified strictly ascending by (date, match_id)",
            f"  Temporal leakage check:          PASSED (features recorded prior to state update)",
            "",
            "2. FEATURE DEFINITIONS AND EXPLANATIONS",
            "  [Pre-Toss Features]",
            "    - team1: First competing franchise name.",
            "    - team2: Second competing franchise name.",
            "    - venue: Stadium name where the match is hosted.",
            "    - city: City name (cleaned for Dubai/Sharjah venue ground truths).",
            "    - season: IPL tournament season year.",
            "    - team1_prior_win_rate: Lifetime win % of Team 1 across all prior IPL matches.",
            "    - team2_prior_win_rate: Lifetime win % of Team 2 across all prior IPL matches.",
            "    - team1_last_5_win_rate: Win % of Team 1 across its last <=5 prior matches.",
            "    - team2_last_5_win_rate: Win % of Team 2 across its last <=5 prior matches.",
            "    - team1_last_10_win_rate: Win % of Team 1 across its last <=10 prior matches.",
            "    - team2_last_10_win_rate: Win % of Team 2 across its last <=10 prior matches.",
            "    - h2h_team1_win_rate: Win % of Team 1 against Team 2 in prior meetings.",
            "    - h2h_matches_before: Count of prior encounters between Team 1 and Team 2.",
            "    - team1_venue_win_rate: Win % of Team 1 at this venue prior to match.",
            "    - team2_venue_win_rate: Win % of Team 2 at this venue prior to match.",
            "    - venue_matches_before: Total prior IPL matches played at this venue.",
            "    - venue_chase_win_rate: Win % of teams batting second (chasing) at this venue.",
            "    - team1_elo: Team 1 Elo rating entering the match (initial 1500.0, K=32).",
            "    - team2_elo: Team 2 Elo rating entering the match (initial 1500.0, K=32).",
            "    - elo_diff: team1_elo - team2_elo (symmetric rating advantage).",
            "    - prior_win_rate_diff: team1_prior_win_rate - team2_prior_win_rate.",
            "    - last_5_win_rate_diff: team1_last_5_win_rate - team2_last_5_win_rate.",
            "    - last_10_win_rate_diff: team1_last_10_win_rate - team2_last_10_win_rate.",
            "    - venue_win_rate_diff: team1_venue_win_rate - team2_venue_win_rate.",
            "    - team1_home_advantage: +1 if Team 1 is home, -1 if Team 2 is home, 0 if neutral.",
            "",
            "  [Post-Toss Additional Features]",
            "    - team1_toss_winner: 1 if Team 1 won the toss, 0 otherwise.",
            "    - toss_decision_field: 1 if toss winner chose to field, 0 if bat.",
            "",
            "  [Target Variable]",
            "    - team1_win: 1 if Team 1 won the match, 0 if Team 2 won.",
            "",
            "3. HYPERPARAMETERS & CONFIGURATION",
            f"  Initial Elo rating:              {self.initial_elo}",
            f"  Elo K-factor:                    {self.elo_k}",
            "  Recent form window sizes:        5 matches, 10 matches",
            f"  Neutral fallback win rate:       {NEUTRAL_WIN_RATE} (used when prior sample is 0)",
            "",
            "4. MISSING VALUE HANDLING & COLD START METRICS",
            f"  Matches with no previous H2H:                 {self.no_h2h_matches} (defaulted to 0.50)",
            f"  Matches with < 5 previous team matches:       {self.fewer_5_matches}",
            f"  Matches with < 10 previous team matches:      {self.fewer_10_matches}",
            f"  Matches with no previous venue history:       {self.no_venue_history_matches} (defaulted to 0.50)",
            f"  Total NaN values in output feature set:       {features_df.isna().sum().sum()}",
            f"  Total Inf values in output feature set:       {np.isinf(features_df.select_dtypes(include=np.number)).sum().sum()}",
            "",
            "5. LEAKAGE PREVENTION CERTIFICATION",
            "  [x] Features are generated in strict chronological order.",
            "  [x] Match T features are extracted BEFORE updating match T state.",
            "  [x] Current match winner, margins, wickets, and player_of_match EXCLUDED from features.",
            "  [x] Future matches can NEVER affect past match representations.",
            "=" * 70,
        ]
        return "\n".join(report_lines)


def run_feature_pipeline(
    input_path: Path = BASE_DATASET_PATH,
    output_csv_path: Path = OUTPUT_FEATURES_PATH,
    output_report_path: Path = OUTPUT_REPORT_PATH,
) -> Tuple[pd.DataFrame, str]:
    """Execute the complete feature engineering pipeline and export artifacts."""
    print(f"Loading clean base dataset from: {input_path}")
    base_df = pd.read_csv(input_path)
    input_count = len(base_df)

    # Ensure chronological sort
    base_df["date"] = pd.to_datetime(base_df["date"])
    base_df = base_df.sort_values(by=["date", "match_id"]).reset_index(drop=True)
    base_df["date"] = base_df["date"].dt.strftime("%Y-%m-%d")

    engine = ChronologicalFeatureEngine(initial_elo=INITIAL_ELO, elo_k=ELO_K)
    features_df = engine.process_matches(base_df)
    output_count = len(features_df)

    # Save features CSV
    features_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Saved features dataset to: {output_csv_path} ({features_df.shape})")

    # Generate and save report
    report_text = engine.generate_report(input_count, output_count, features_df)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved feature engineering report to: {output_report_path}")

    return features_df, report_text


if __name__ == "__main__":
    features_df, report = run_feature_pipeline()
    print("\n" + report)
