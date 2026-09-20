"""
prediction_engine.py
====================
Inference Engine for IPL Oracle Predictions.
Loads trained scikit-learn pipelines directly via joblib, validates input fixtures,
assembles leakage-safe pre-match feature vectors using the verified historical state,
and outputs model-estimated win probabilities via predict_proba().
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

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.feature_engineering import HOME_CITIES, INITIAL_ELO, ELO_K, VENUE_CITY_MAPPING
from ml.parse_cricsheet import TEAM_NAME_MAPPING, TEAM_CODE_MAPPING

# Model file paths
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
BASE_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "ipl_matches_base.csv"

PRE_TOSS_MODEL_PATH = MODELS_DIR / "pre_toss_improved_histgb.joblib"
POST_TOSS_MODEL_PATH = MODELS_DIR / "post_toss_improved_histgb.joblib"

# Inverse team code mapping (e.g. 'MI' -> 'Mumbai Indians')
CODE_TO_TEAM_MAPPING: Dict[str, str] = {
    code.upper(): name for name, code in TEAM_CODE_MAPPING.items()
}
# Add canonical names to code mapping
CODE_TO_TEAM_MAPPING.update({
    "RCB": "Royal Challengers Bangalore",
    "DC": "Delhi Capitals",
    "PBKS": "Punjab Kings",
    "RPS": "Rising Pune Supergiant",
})

# All known normalized franchises
KNOWN_TEAMS: Set[str] = {
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Gujarat Titans",
    "Lucknow Super Giants",
    # Historical / Defunct
    "Deccan Chargers",
    "Pune Warriors",
    "Gujarat Lions",
    "Rising Pune Supergiant",
    "Kochi Tuskers Kerala",
}

# Ground truth venue to primary city lookup
VENUE_TO_CITY: Dict[str, str] = {
    "Wankhede Stadium": "Mumbai",
    "Wankhede Stadium, Mumbai": "Mumbai",
    "Brabourne Stadium": "Mumbai",
    "Brabourne Stadium, Mumbai": "Mumbai",
    "Dr DY Patil Sports Academy": "Mumbai",
    "Dr DY Patil Sports Academy, Mumbai": "Mumbai",
    "MA Chidambaram Stadium": "Chennai",
    "MA Chidambaram Stadium, Chepauk": "Chennai",
    "MA Chidambaram Stadium, Chepauk, Chennai": "Chennai",
    "Eden Gardens": "Kolkata",
    "Eden Gardens, Kolkata": "Kolkata",
    "M Chinnaswamy Stadium": "Bengaluru",
    "M. Chinnaswamy Stadium": "Bengaluru",
    "M Chinnaswamy Stadium, Bengaluru": "Bengaluru",
    "Arun Jaitley Stadium": "Delhi",
    "Arun Jaitley Stadium, Delhi": "Delhi",
    "Feroz Shah Kotla": "Delhi",
    "Rajiv Gandhi International Stadium": "Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal": "Hyderabad",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "Hyderabad",
    "Sawai Mansingh Stadium": "Jaipur",
    "Sawai Mansingh Stadium, Jaipur": "Jaipur",
    "Punjab Cricket Association Stadium, Mohali": "Mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "Mohali",
    "Punjab Cricket Association IS Bindra Stadium": "Mohali",
    "Narendra Modi Stadium": "Ahmedabad",
    "Narendra Modi Stadium, Ahmedabad": "Ahmedabad",
    "Sardar Patel Stadium, Motera": "Ahmedabad",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "Lucknow",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "Lucknow",
    "Dubai International Cricket Stadium": "Dubai",
    "Sharjah Cricket Stadium": "Sharjah",
    "Zayed Cricket Stadium, Abu Dhabi": "Abu Dhabi",
    "Sheikh Zayed Stadium": "Abu Dhabi",
    "Himachal Pradesh Cricket Association Stadium": "Dharamsala",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala": "Dharamsala",
    "Maharashtra Cricket Association Stadium": "Pune",
    "Maharashtra Cricket Association Stadium, Pune": "Pune",
    "Subrata Roy Sahara Stadium": "Pune",
    "Barabati Stadium": "Cuttack",
    "JSCA International Stadium Complex": "Ranchi",
    "Shaheed Veer Narayan Singh International Stadium": "Raipur",
    "Holkar Cricket Stadium": "Indore",
    "Green Park": "Kanpur",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "Visakhapatnam",
    "ACA-VDCA Stadium": "Visakhapatnam",
}


def normalize_team(input_str: str) -> str:
    """Normalize user input team name or code into canonical franchise name."""
    if not input_str or not isinstance(input_str, str):
        return ""
    cleaned = input_str.strip()
    upper = cleaned.upper()
    if upper in CODE_TO_TEAM_MAPPING:
        return CODE_TO_TEAM_MAPPING[upper]
    mapped = TEAM_NAME_MAPPING.get(cleaned, cleaned)
    return mapped


class PredictionEngine:
    """
    Production-ready Inference Engine for IPL Oracle Predictions.
    Loads trained scikit-learn pipelines directly via joblib and constructs
    exact feature representations using historical state.
    """

    def __init__(
        self,
        pre_toss_model_path: Path = PRE_TOSS_MODEL_PATH,
        post_toss_model_path: Path = POST_TOSS_MODEL_PATH,
        base_dataset_path: Path = BASE_DATASET_PATH,
    ) -> None:
        self.pre_toss_model_path = pre_toss_model_path
        self.post_toss_model_path = post_toss_model_path
        self.base_dataset_path = base_dataset_path

        self.pre_toss_model: Optional[Pipeline] = None
        self.post_toss_model: Optional[Pipeline] = None

        self.pre_toss_expected_features: List[str] = []
        self.post_toss_expected_features: List[str] = []

        # Historical state accumulators
        self.team_history: Dict[str, List[int]] = defaultdict(list)
        self.team_season_history: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.team_elo_history: Dict[str, List[float]] = defaultdict(lambda: [INITIAL_ELO])
        self.elo_ratings: Dict[str, float] = defaultdict(lambda: INITIAL_ELO)
        self.h2h_history: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.venue_team_history: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.venue_matches_count: Dict[str, int] = defaultdict(int)
        self.venue_chase_history: Dict[str, List[int]] = defaultdict(list)
        self.known_venues: Set[str] = set()
        self.latest_season: str = "2026"

        self.load_models()
        self.build_historical_state()

    def load_models(self) -> None:
        """Load trained models from joblib files."""
        if not self.pre_toss_model_path.exists():
            raise FileNotFoundError(f"Pre-toss model not found at {self.pre_toss_model_path}")
        if not self.post_toss_model_path.exists():
            raise FileNotFoundError(f"Post-toss model not found at {self.post_toss_model_path}")

        self.pre_toss_model = joblib.load(self.pre_toss_model_path)
        self.post_toss_model = joblib.load(self.post_toss_model_path)

        # Read feature names expected by pipelines
        self.pre_toss_expected_features = list(self.pre_toss_model.feature_names_in_)
        self.post_toss_expected_features = list(self.post_toss_model.feature_names_in_)

    def build_historical_state(self) -> None:
        """Build historical accumulators strictly from the processed base dataset."""
        if not self.base_dataset_path.exists():
            raise FileNotFoundError(f"Base dataset not found at {self.base_dataset_path}")

        base_df = pd.read_csv(self.base_dataset_path)
        base_df["date"] = pd.to_datetime(base_df["date"])
        base_df = base_df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

        for _, row in base_df.iterrows():
            season = str(row["season"])
            self.latest_season = season
            venue = str(row["venue"])
            self.known_venues.add(venue)
            t1 = str(row["team1"])
            t2 = str(row["team2"])
            tw = str(row["toss_winner"])
            td = str(row["toss_decision"]).lower().strip()
            winner = str(row["winner"])

            t1_won = (winner == t1)
            t2_won = (winner == t2)

            self.team_history[t1].append(1 if t1_won else 0)
            self.team_history[t2].append(1 if t2_won else 0)
            self.team_season_history[season][t1].append(1 if t1_won else 0)
            self.team_season_history[season][t2].append(1 if t2_won else 0)

            h2h_key = tuple(sorted([t1, t2]))
            self.h2h_history[h2h_key].append(winner)

            self.venue_team_history[(venue, t1)].append(1 if t1_won else 0)
            self.venue_team_history[(venue, t2)].append(1 if t2_won else 0)
            self.venue_matches_count[venue] += 1

            # Chasing outcome
            chasing_team = t1 if (tw == t1 and td == "field") or (tw == t2 and td == "bat") else t2
            chase_won = 1 if winner == chasing_team else 0
            self.venue_chase_history[venue].append(chase_won)

            # Elo update
            r1 = self.elo_ratings[t1]
            r2 = self.elo_ratings[t2]
            e1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
            e2 = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))
            s1 = 1.0 if t1_won else 0.0
            s2 = 1.0 - s1
            self.elo_ratings[t1] = r1 + ELO_K * (s1 - e1)
            self.elo_ratings[t2] = r2 + ELO_K * (s2 - e2)
            self.team_elo_history[t1].append(self.elo_ratings[t1])
            self.team_elo_history[t2].append(self.elo_ratings[t2])

    def validate_input(
        self,
        team1_raw: str,
        team2_raw: str,
        venue_raw: str,
        city_raw: Optional[str] = None,
        toss_winner_raw: Optional[str] = None,
        toss_decision_raw: Optional[str] = None,
    ) -> Tuple[str, str, str, str, Optional[str], Optional[str]]:
        """Validate and normalize user input fixture parameters."""
        if not team1_raw or not isinstance(team1_raw, str) or not team1_raw.strip():
            raise ValueError("Parameter 'team1' is required and must be a non-empty string.")
        if not team2_raw or not isinstance(team2_raw, str) or not team2_raw.strip():
            raise ValueError("Parameter 'team2' is required and must be a non-empty string.")
        if not venue_raw or not isinstance(venue_raw, str) or not venue_raw.strip():
            raise ValueError("Parameter 'venue' is required and must be a non-empty string.")

        t1 = normalize_team(team1_raw)
        t2 = normalize_team(team2_raw)

        if not t1 or t1 not in KNOWN_TEAMS:
            raise ValueError(
                f"Unknown team '{team1_raw}'. Must be one of the known IPL franchises: {sorted(list(KNOWN_TEAMS))}"
            )
        if not t2 or t2 not in KNOWN_TEAMS:
            raise ValueError(
                f"Unknown team '{team2_raw}'. Must be one of the known IPL franchises: {sorted(list(KNOWN_TEAMS))}"
            )

        if t1 == t2:
            raise ValueError(f"team1 and team2 must be different teams. Both specified as '{t1}'.")

        venue = venue_raw.strip()

        # Derive city if missing
        city = city_raw.strip() if (city_raw and isinstance(city_raw, str) and city_raw.strip()) else ""
        if not city:
            city = VENUE_CITY_MAPPING.get(venue) or VENUE_TO_CITY.get(venue, "")

        # Toss validation
        tw: Optional[str] = None
        td: Optional[str] = None

        if toss_winner_raw or toss_decision_raw:
            if not toss_winner_raw:
                raise ValueError("toss_winner must be specified when toss_decision is provided.")
            if not toss_decision_raw:
                raise ValueError("toss_decision must be specified when toss_winner is provided ('bat' or 'field').")

            tw = normalize_team(toss_winner_raw)
            if tw not in (t1, t2):
                raise ValueError(f"toss_winner '{toss_winner_raw}' must be either team1 ('{t1}') or team2 ('{t2}').")

            td_clean = toss_decision_raw.strip().lower()
            if td_clean not in ("bat", "field"):
                raise ValueError(f"toss_decision '{toss_decision_raw}' must be either 'bat' or 'field'.")
            td = td_clean

        return t1, t2, venue, city, tw, td

    def build_feature_dataframe(
        self,
        team1: str,
        team2: str,
        venue: str,
        city: str,
        toss_winner: Optional[str] = None,
        toss_decision: Optional[str] = None,
    ) -> pd.DataFrame:
        """Construct exact pre-match feature DataFrame matching model schema."""
        # 1. Lifetime win rate
        t1_prior = self.team_history[team1]
        t2_prior = self.team_history[team2]
        t1_p_win = np.mean(t1_prior) if t1_prior else 0.50
        t2_p_win = np.mean(t2_prior) if t2_prior else 0.50
        p_win_diff = t1_p_win - t2_p_win

        # 2. Form (last 3, 5, 10)
        t1_last3 = np.mean(t1_prior[-3:]) if t1_prior else 0.50
        t2_last3 = np.mean(t2_prior[-3:]) if t2_prior else 0.50
        last3_diff = t1_last3 - t2_last3

        t1_last5 = np.mean(t1_prior[-5:]) if t1_prior else 0.50
        t2_last5 = np.mean(t2_prior[-5:]) if t2_prior else 0.50
        last5_diff = t1_last5 - t2_last5

        t1_last10 = np.mean(t1_prior[-10:]) if t1_prior else 0.50
        t2_last10 = np.mean(t2_prior[-10:]) if t2_prior else 0.50
        last10_diff = t1_last10 - t2_last10

        # 3. Current in-season performance
        t1_season = self.team_season_history[self.latest_season][team1]
        t2_season = self.team_season_history[self.latest_season][team2]
        t1_season_wr = np.mean(t1_season) if t1_season else 0.50
        t2_season_wr = np.mean(t2_season) if t2_season else 0.50
        season_wr_diff = t1_season_wr - t2_season_wr
        t1_season_games = len(t1_season)
        t2_season_games = len(t2_season)

        # 4. Head to Head
        h2h_key = tuple(sorted([team1, team2]))
        h2h_past = self.h2h_history[h2h_key]
        h2h_count = len(h2h_past)
        if h2h_count == 0:
            h2h_t1_wr = 0.50
        else:
            h2h_t1_wr = sum(1 for w in h2h_past if w == team1) / h2h_count

        # 5. Venue performance
        v_t1 = self.venue_team_history[(venue, team1)]
        v_t2 = self.venue_team_history[(venue, team2)]
        t1_v_win = np.mean(v_t1) if v_t1 else 0.50
        t2_v_win = np.mean(v_t2) if v_t2 else 0.50
        v_win_diff = t1_v_win - t2_v_win
        v_matches_before = self.venue_matches_count[venue]

        v_chase = self.venue_chase_history[venue]
        venue_chase_wr = np.mean(v_chase) if v_chase else 0.50

        # 6. Elo & Elo dynamics
        t1_elo = self.elo_ratings[team1]
        t2_elo = self.elo_ratings[team2]
        elo_diff = t1_elo - t2_elo
        elo_prob_t1 = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        t1_elo_hist = self.team_elo_history[team1]
        t2_elo_hist = self.team_elo_history[team2]
        t1_elo_mom = t1_elo - t1_elo_hist[-6] if len(t1_elo_hist) >= 6 else (t1_elo - t1_elo_hist[0])
        t2_elo_mom = t2_elo - t2_elo_hist[-6] if len(t2_elo_hist) >= 6 else (t2_elo - t2_elo_hist[0])
        elo_mom_diff = t1_elo_mom - t2_elo_mom

        # 7. Home advantage
        t1_is_home = 1 if city in HOME_CITIES.get(team1, set()) else 0
        t2_is_home = 1 if city in HOME_CITIES.get(team2, set()) else 0
        t1_home_adv = t1_is_home - t2_is_home

        row_dict: Dict[str, Any] = {
            "team1": team1,
            "team2": team2,
            "venue": venue,
            "city": city,
            "team1_prior_win_rate": t1_p_win,
            "team2_prior_win_rate": t2_p_win,
            "prior_win_rate_diff": p_win_diff,
            "team1_last_5_win_rate": t1_last5,
            "team2_last_5_win_rate": t2_last5,
            "last_5_win_rate_diff": last5_diff,
            "team1_last_10_win_rate": t1_last10,
            "team2_last_10_win_rate": t2_last10,
            "last_10_win_rate_diff": last10_diff,
            "h2h_team1_win_rate": h2h_t1_wr,
            "h2h_matches_before": h2h_count,
            "team1_venue_win_rate": t1_v_win,
            "team2_venue_win_rate": t2_v_win,
            "venue_win_rate_diff": v_win_diff,
            "venue_matches_before": v_matches_before,
            "venue_chase_win_rate": venue_chase_wr,
            "team1_elo": t1_elo,
            "team2_elo": t2_elo,
            "elo_diff": elo_diff,
            "team1_home_advantage": t1_home_adv,
            "elo_prob_t1": elo_prob_t1,
            "team1_last_3_win_rate": t1_last3,
            "team2_last_3_win_rate": t2_last3,
            "last_3_win_rate_diff": last3_diff,
            "team1_elo_momentum": t1_elo_mom,
            "team2_elo_momentum": t2_elo_mom,
            "elo_momentum_diff": elo_mom_diff,
            "team1_season_win_rate": t1_season_wr,
            "team2_season_win_rate": t2_season_wr,
            "season_win_rate_diff": season_wr_diff,
            "team1_season_games": t1_season_games,
            "team2_season_games": t2_season_games,
        }

        # Post-toss features if toss is provided
        if toss_winner and toss_decision:
            t1_toss_winner = 1 if toss_winner == team1 else 0
            toss_decision_field = 1 if toss_decision == "field" else 0
            if toss_winner == team1:
                t1_chasing = 1 if toss_decision == "field" else 0
            else:
                t1_chasing = 1 if toss_decision == "bat" else 0
            t1_chase_venue_adv = (1 if t1_chasing else -1) * (venue_chase_wr - 0.50)

            row_dict["team1_toss_winner"] = t1_toss_winner
            row_dict["toss_decision_field"] = toss_decision_field
            row_dict["team1_is_chasing"] = t1_chasing
            row_dict["team1_chase_venue_adv"] = t1_chase_venue_adv

        # Return DataFrame with expected column order
        cols = self.post_toss_expected_features if (toss_winner and toss_decision) else self.pre_toss_expected_features
        return pd.DataFrame([row_dict], columns=cols)

    def predict(
        self,
        team1: str,
        team2: str,
        venue: str,
        city: Optional[str] = None,
        toss_winner: Optional[str] = None,
        toss_decision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate ML win prediction and probabilities for a fixture.

        Args:
            team1: Name or code of Team 1 (e.g. 'Mumbai Indians' or 'MI')
            team2: Name or code of Team 2 (e.g. 'Chennai Super Kings' or 'CSK')
            venue: Stadium name
            city: Optional city location
            toss_winner: Optional toss winner ('team1' or 'team2' name)
            toss_decision: Optional toss decision ('bat' or 'field')

        Returns:
            JSON-serializable prediction response dictionary.
        """
        # 1. Validation and normalization
        t1, t2, v, c, tw, td = self.validate_input(
            team1_raw=team1,
            team2_raw=team2,
            venue_raw=venue,
            city_raw=city,
            toss_winner_raw=toss_winner,
            toss_decision_raw=toss_decision,
        )

        is_post_toss = bool(tw and td)
        model_type = "post_toss" if is_post_toss else "pre_toss"
        pipeline = self.post_toss_model if is_post_toss else self.pre_toss_model

        if pipeline is None:
            raise RuntimeError(f"{model_type} model pipeline is not loaded.")

        # 2. Build feature vector
        features_df = self.build_feature_dataframe(
            team1=t1,
            team2=t2,
            venue=v,
            city=c,
            toss_winner=tw,
            toss_decision=td,
        )

        # 3. Model predict_proba()
        probs = pipeline.predict_proba(features_df)[0]
        prob_t2 = float(probs[0])
        prob_t1 = float(probs[1])

        # Verify probability sum
        prob_sum = prob_t1 + prob_t2
        if abs(prob_sum - 1.0) > 1e-4:
            prob_t1 /= prob_sum
            prob_t2 /= prob_sum

        # Predicted winner
        pred_winner = t1 if prob_t1 >= 0.50 else t2

        # Confidence categorization matching IPL Oracle standards
        diff = abs(prob_t1 - prob_t2)
        if diff >= 0.20:
            confidence = "High"
        elif diff >= 0.06:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Build detailed analytics
        row = features_df.iloc[0]
        details = {
            "team1_elo": round(float(row["team1_elo"]), 1),
            "team2_elo": round(float(row["team2_elo"]), 1),
            "elo_diff": round(float(row["elo_diff"]), 1),
            "h2h_matches_before": int(row["h2h_matches_before"]),
            "h2h_team1_win_rate": round(float(row["h2h_team1_win_rate"]), 4),
            "team1_prior_win_rate": round(float(row["team1_prior_win_rate"]), 4),
            "team2_prior_win_rate": round(float(row["team2_prior_win_rate"]), 4),
            "team1_venue_win_rate": round(float(row["team1_venue_win_rate"]), 4),
            "team2_venue_win_rate": round(float(row["team2_venue_win_rate"]), 4),
            "venue_matches_before": int(row["venue_matches_before"]),
            "venue_chase_win_rate": round(float(row["venue_chase_win_rate"]), 4),
            "team1_home_advantage": int(row["team1_home_advantage"]),
        }

        if is_post_toss:
            details["toss_winner"] = tw
            details["toss_decision"] = td
            details["team1_is_chasing"] = bool(row["team1_is_chasing"])
            details["team1_chase_venue_adv"] = round(float(row["team1_chase_venue_adv"]), 4)

        return {
            "model": model_type,
            "team1": t1,
            "team2": t2,
            "venue": v,
            "city": c,
            "team1_win_probability": round(prob_t1, 4),
            "team2_win_probability": round(prob_t2, 4),
            "predicted_winner": pred_winner,
            "confidence": confidence,
            "details": details,
        }


if __name__ == "__main__":
    engine = PredictionEngine()
    print("\nTesting Pre-Toss Prediction (MI vs CSK at Wankhede):")
    res_pre = engine.predict(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
    )
    print(res_pre)

    print("\nTesting Post-Toss Prediction (MI vs CSK at Wankhede, MI wins toss, bowls):")
    res_post = engine.predict(
        team1="MI",
        team2="CSK",
        venue="Wankhede Stadium",
        toss_winner="MI",
        toss_decision="field",
    )
    print(res_post)
