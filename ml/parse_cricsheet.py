"""
parse_cricsheet.py
==================
Parser for Cricsheet IPL historical CSV match files (Original Format v1.8.0).
Reads CSV files directly from the extracted Cricsheet dataset, parses 'version',
'info', and 'ball' records, normalizes franchise names and dates, computes
innings and match-level aggregate statistics, and provides clean data structures
for downstream dataset construction and feature engineering.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default location of the extracted Cricsheet IPL dataset
DEFAULT_DATASET_DIR = r"C:\Users\ANUKTAA\Downloads\ipl_csv"

# Canonical franchise name normalization mapping
# Preserves active 10 franchises consistent with ipl.html UI and official names
TEAM_NAME_MAPPING: Dict[str, str] = {
    # Delhi Daredevils rebrand
    "Delhi Daredevils": "Delhi Capitals",
    # Kings XI Punjab rebrand
    "Kings XI Punjab": "Punjab Kings",
    # Royal Challengers Bengaluru / Bangalore alignment (matches ipl.html)
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    # Rising Pune Supergiants spelling variation
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Standard short codes for IPL franchises
TEAM_CODE_MAPPING: Dict[str, str] = {
    "Mumbai Indians": "MI",
    "Chennai Super Kings": "CSK",
    "Royal Challengers Bangalore": "RCB",
    "Royal Challengers Bengaluru": "RCB",
    "Kolkata Knight Riders": "KKR",
    "Sunrisers Hyderabad": "SRH",
    "Delhi Capitals": "DC",
    "Delhi Daredevils": "DC",
    "Punjab Kings": "PBKS",
    "Kings XI Punjab": "PBKS",
    "Rajasthan Royals": "RR",
    "Gujarat Titans": "GT",
    "Lucknow Super Giants": "LSG",
    # Defunct franchises
    "Deccan Chargers": "DCG",
    "Pune Warriors": "PWI",
    "Gujarat Lions": "GL",
    "Kochi Tuskers Kerala": "KTK",
    "Rising Pune Supergiant": "RPS",
    "Rising Pune Supergiants": "RPS",
}

# Positional field names for Cricsheet 'ball' records (21 columns)
BALL_COLUMNS: List[str] = [
    "row_type",
    "innings",
    "delivery",
    "batting_team",
    "batter",
    "non_striker",
    "bowler",
    "runs_off_bat",
    "extras",
    "wides",
    "noballs",
    "byes",
    "legbyes",
    "penalty",
    "wicket_type",
    "player_dismissed",
    "actual_delivery",
    "non_boundary",
    "fielder_1",
    "fielder_2",
    "fielder_3",
]


def normalize_team_name(team_name: Optional[str]) -> Optional[str]:
    """Normalize historical team names into consistent franchise names."""
    if not team_name:
        return None
    cleaned = team_name.strip()
    return TEAM_NAME_MAPPING.get(cleaned, cleaned)


def get_team_code(team_name: Optional[str]) -> str:
    """Return standard 2-4 letter team acronym/code."""
    if not team_name:
        return ""
    cleaned = team_name.strip()
    return TEAM_CODE_MAPPING.get(cleaned, "".join(w[0] for w in cleaned.split()).upper())


def normalize_date(date_str: Optional[str]) -> str:
    """
    Normalize dates from 'YYYY/MM/DD' or other representations to 'YYYY-MM-DD'.
    """
    if not date_str:
        return ""
    cleaned = date_str.strip().replace("/", "-")
    # Validate and format
    try:
        dt = datetime.strptime(cleaned[:10], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return cleaned[:10]


@dataclass
class InningsStats:
    """Aggregated statistics for a single innings."""
    innings: int
    batting_team: str
    batting_team_normalized: str
    runs: int = 0
    wickets: int = 0
    legal_balls: int = 0
    total_deliveries: int = 0
    fours: int = 0
    sixes: int = 0
    extras: int = 0
    wides: int = 0
    noballs: int = 0
    byes: int = 0
    legbyes: int = 0
    penalty: int = 0

    @property
    def overs_formatted(self) -> str:
        """Format legal balls into standard cricket overs (e.g. 19.4)."""
        return f"{self.legal_balls // 6}.{self.legal_balls % 6}"


@dataclass
class ParsedMatch:
    """Parsed match-level and delivery-level data representation."""
    match_id: str
    season: str
    date: str
    venue: str
    city: str
    team1: str
    team2: str
    team1_orig: str
    team2_orig: str
    toss_winner: str
    toss_winner_orig: str
    toss_decision: str
    winner: Optional[str]
    winner_orig: Optional[str]
    eliminator: Optional[str]
    outcome: Optional[str]
    match_status: str  # 'completed', 'tie_with_eliminator', 'no_result'
    team1_win: Optional[int]  # 1, 0, or None for no result
    player_of_match: Optional[str]
    method: Optional[str] = None  # DLS if rain affected
    winner_runs: Optional[int] = None
    winner_wickets: Optional[int] = None
    target_runs: Optional[int] = None
    target_overs: Optional[float] = None
    playing_xi_orig: Dict[str, List[str]] = field(default_factory=dict)
    playing_xi_norm: Dict[str, List[str]] = field(default_factory=dict)
    innings_stats: Dict[int, InningsStats] = field(default_factory=dict)
    ball_records: List[Dict[str, Any]] = field(default_factory=list)
    raw_info: Dict[str, Any] = field(default_factory=dict)
    file_path: str = ""

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a tabular dictionary suitable for DataFrame creation."""
        inn1 = self.innings_stats.get(1)
        inn2 = self.innings_stats.get(2)
        return {
            "match_id": self.match_id,
            "season": self.season,
            "date": self.date,
            "venue": self.venue,
            "city": self.city,
            "team1": self.team1,
            "team2": self.team2,
            "team1_orig": self.team1_orig,
            "team2_orig": self.team2_orig,
            "toss_winner": self.toss_winner,
            "toss_winner_orig": self.toss_winner_orig,
            "toss_decision": self.toss_decision,
            "winner": self.winner,
            "winner_orig": self.winner_orig,
            "eliminator": self.eliminator,
            "outcome": self.outcome,
            "match_status": self.match_status,
            "team1_win": self.team1_win,
            "player_of_match": self.player_of_match,
            "method": self.method,
            "inn1_runs": inn1.runs if inn1 else None,
            "inn1_wickets": inn1.wickets if inn1 else None,
            "inn1_balls": inn1.legal_balls if inn1 else None,
            "inn1_fours": inn1.fours if inn1 else None,
            "inn1_sixes": inn1.sixes if inn1 else None,
            "inn1_extras": inn1.extras if inn1 else None,
            "inn2_runs": inn2.runs if inn2 else None,
            "inn2_wickets": inn2.wickets if inn2 else None,
            "inn2_balls": inn2.legal_balls if inn2 else None,
            "inn2_fours": inn2.fours if inn2 else None,
            "inn2_sixes": inn2.sixes if inn2 else None,
            "inn2_extras": inn2.extras if inn2 else None,
        }


def compute_innings_statistics(ball_records: List[Dict[str, Any]]) -> Dict[int, InningsStats]:
    """Compute aggregated innings statistics from ball-by-ball delivery records."""
    stats_by_innings: Dict[int, InningsStats] = {}

    for ball in ball_records:
        inn = ball.get("innings", 1)
        if inn not in stats_by_innings:
            bat_team = ball.get("batting_team", "")
            stats_by_innings[inn] = InningsStats(
                innings=inn,
                batting_team=bat_team,
                batting_team_normalized=normalize_team_name(bat_team) or bat_team,
            )

        inn_stat = stats_by_innings[inn]
        inn_stat.total_deliveries += 1

        runs_bat = ball.get("runs_off_bat", 0)
        extras = ball.get("extras", 0)
        wides = ball.get("wides", 0)
        noballs = ball.get("noballs", 0)
        byes = ball.get("byes", 0)
        legbyes = ball.get("legbyes", 0)
        penalty = ball.get("penalty", 0)

        inn_stat.runs += (runs_bat + extras)
        inn_stat.extras += extras
        inn_stat.wides += wides
        inn_stat.noballs += noballs
        inn_stat.byes += byes
        inn_stat.legbyes += legbyes
        inn_stat.penalty += penalty

        # Legal delivery check: wides and noballs are illegal deliveries
        if wides == 0 and noballs == 0:
            inn_stat.legal_balls += 1

        # Boundaries (non_boundary flag means runs were completed via running/overthrow)
        if runs_bat == 4 and not ball.get("non_boundary"):
            inn_stat.fours += 1
        elif runs_bat == 6 and not ball.get("non_boundary"):
            inn_stat.sixes += 1

        # Wickets (retired hurt is not credited as a wicket loss)
        w_type = ball.get("wicket_type", "")
        if w_type and w_type.lower() != "retired hurt":
            inn_stat.wickets += 1

    return stats_by_innings


def parse_cricsheet_csv(file_path: str, parse_balls: bool = True) -> ParsedMatch:
    """
    Parse a single Cricsheet IPL CSV file into a structured ParsedMatch instance.

    Args:
        file_path: Absolute or relative path to the Cricsheet CSV file.
        parse_balls: Whether to parse delivery rows and compute ball statistics.

    Returns:
        ParsedMatch object containing extracted metadata and statistics.
    """
    raw_info: Dict[str, Any] = {}
    teams_list: List[str] = []
    dates_list: List[str] = []
    players_orig: Dict[str, List[str]] = {}
    players_norm: Dict[str, List[str]] = {}
    ball_records: List[Dict[str, Any]] = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            record_type = row[0].strip()

            if record_type == "version":
                raw_info["version"] = row[1].strip() if len(row) > 1 else ""

            elif record_type == "info":
                if len(row) < 3:
                    continue
                key = row[1].strip()
                val = row[2].strip()

                if key == "team":
                    teams_list.append(val)
                elif key == "date":
                    dates_list.append(val)
                elif key == "player":
                    if len(row) >= 4:
                        p_team = row[2].strip()
                        p_name = row[3].strip()
                        players_orig.setdefault(p_team, []).append(p_name)
                        norm_team = normalize_team_name(p_team) or p_team
                        players_norm.setdefault(norm_team, []).append(p_name)
                elif key in ("umpire", "reserve_umpire", "tv_umpire"):
                    raw_info.setdefault(key, []).append(val)
                elif key == "registry":
                    pass  # Registry IDs not needed for match modeling
                else:
                    raw_info[key] = val

            elif record_type == "ball" and parse_balls:
                # Map positional fields
                ball_data: Dict[str, Any] = {
                    "row_type": "ball",
                    "innings": int(row[1]) if len(row) > 1 and row[1].isdigit() else 1,
                    "delivery": row[2] if len(row) > 2 else "",
                    "batting_team": row[3] if len(row) > 3 else "",
                    "batter": row[4] if len(row) > 4 else "",
                    "non_striker": row[5] if len(row) > 5 else "",
                    "bowler": row[6] if len(row) > 6 else "",
                    "runs_off_bat": int(row[7]) if len(row) > 7 and row[7].isdigit() else 0,
                    "extras": int(row[8]) if len(row) > 8 and row[8].isdigit() else 0,
                    "wides": int(row[9]) if len(row) > 9 and row[9].isdigit() else 0,
                    "noballs": int(row[10]) if len(row) > 10 and row[10].isdigit() else 0,
                    "byes": int(row[11]) if len(row) > 11 and row[11].isdigit() else 0,
                    "legbyes": int(row[12]) if len(row) > 12 and row[12].isdigit() else 0,
                    "penalty": int(row[13]) if len(row) > 13 and row[13].isdigit() else 0,
                    "wicket_type": row[14] if len(row) > 14 else "",
                    "player_dismissed": row[15] if len(row) > 15 else "",
                    "actual_delivery": row[16] if len(row) > 16 else "",
                    "non_boundary": row[17] == "true" if len(row) > 17 else False,
                    "fielder_1": row[18] if len(row) > 18 else "",
                    "fielder_2": row[19] if len(row) > 19 else "",
                    "fielder_3": row[20] if len(row) > 20 else "",
                }
                ball_records.append(ball_data)

    # Match ID fallback to file stem if not in info
    match_id = raw_info.get("match_id") or Path(file_path).stem
    season = raw_info.get("season", "")

    # Date normalization: take primary match date
    primary_date = dates_list[0] if dates_list else raw_info.get("date", "")
    date_normalized = normalize_date(primary_date)

    venue = raw_info.get("venue", "")
    city = raw_info.get("city", "")

    # Teams
    team1_orig = teams_list[0] if len(teams_list) > 0 else ""
    team2_orig = teams_list[1] if len(teams_list) > 1 else ""
    team1 = normalize_team_name(team1_orig) or team1_orig
    team2 = normalize_team_name(team2_orig) or team2_orig

    # Toss
    toss_winner_orig = raw_info.get("toss_winner", "")
    toss_winner = normalize_team_name(toss_winner_orig) or toss_winner_orig
    toss_decision = raw_info.get("toss_decision", "").lower()

    # Outcome / Winner / Eliminator determination
    outcome = raw_info.get("outcome")
    winner_orig = raw_info.get("winner")
    eliminator = raw_info.get("eliminator")

    if winner_orig:
        winner = normalize_team_name(winner_orig)
        match_status = "completed"
    elif outcome == "tie" and eliminator:
        winner_orig = eliminator
        winner = normalize_team_name(eliminator)
        match_status = "tie_with_eliminator"
    elif outcome == "no result":
        winner_orig = None
        winner = None
        match_status = "no_result"
    else:
        winner_orig = None
        winner = None
        match_status = outcome or "incomplete"

    # Target variable: team1_win
    team1_win: Optional[int] = None
    if winner is not None:
        if winner == team1:
            team1_win = 1
        elif winner == team2:
            team1_win = 0
        else:
            # Check original names in case of partial mapping
            if winner_orig == team1_orig:
                team1_win = 1
            elif winner_orig == team2_orig:
                team1_win = 0

    # Match summary stats from ball rows
    innings_stats = compute_innings_statistics(ball_records) if parse_balls else {}

    # Player of match
    player_of_match = raw_info.get("player_of_match")

    # Winner margin (if available)
    winner_runs = int(raw_info["winner_runs"]) if raw_info.get("winner_runs") and raw_info["winner_runs"].isdigit() else None
    winner_wickets = int(raw_info["winner_wickets"]) if raw_info.get("winner_wickets") and raw_info["winner_wickets"].isdigit() else None

    # Target info (if available)
    target_runs = int(raw_info["target_runs"]) if raw_info.get("target_runs") and raw_info["target_runs"].isdigit() else None
    try:
        target_overs = float(raw_info["target_overs"]) if raw_info.get("target_overs") else None
    except ValueError:
        target_overs = None

    return ParsedMatch(
        match_id=match_id,
        season=season,
        date=date_normalized,
        venue=venue,
        city=city,
        team1=team1,
        team2=team2,
        team1_orig=team1_orig,
        team2_orig=team2_orig,
        toss_winner=toss_winner,
        toss_winner_orig=toss_winner_orig,
        toss_decision=toss_decision,
        winner=winner,
        winner_orig=winner_orig,
        eliminator=eliminator,
        outcome=outcome,
        match_status=match_status,
        team1_win=team1_win,
        player_of_match=player_of_match,
        method=raw_info.get("method"),
        winner_runs=winner_runs,
        winner_wickets=winner_wickets,
        target_runs=target_runs,
        target_overs=target_overs,
        playing_xi_orig=players_orig,
        playing_xi_norm=players_norm,
        innings_stats=innings_stats,
        ball_records=ball_records,
        raw_info=raw_info,
        file_path=file_path,
    )


def parse_all_matches(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    parse_balls: bool = True,
) -> Tuple[List[ParsedMatch], List[Tuple[str, str]]]:
    """
    Parse all Cricsheet CSV match files from the given directory.

    Returns:
        A tuple of (parsed_matches_list, failed_files_list).
    """
    csv_pattern = os.path.join(dataset_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)

    parsed_matches: List[ParsedMatch] = []
    failed_files: List[Tuple[str, str]] = []

    for fpath in csv_files:
        try:
            match = parse_cricsheet_csv(fpath, parse_balls=parse_balls)
            parsed_matches.append(match)
        except Exception as e:
            failed_files.append((fpath, str(e)))

    return parsed_matches, failed_files


if __name__ == "__main__":
    import sys

    dir_to_parse = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_DIR
    print(f"Testing parse_cricsheet on directory: {dir_to_parse}")

    sample_file = os.path.join(dir_to_parse, "1082591.csv")
    if os.path.exists(sample_file):
        print(f"\nParsing single sample match: {sample_file}")
        sample_match = parse_cricsheet_csv(sample_file)
        summary = sample_match.to_summary_dict()
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"  Innings 1 Balls: {sample_match.innings_stats.get(1)}")
        print(f"  Innings 2 Balls: {sample_match.innings_stats.get(2)}")
    else:
        print(f"Sample file {sample_file} not found.")
