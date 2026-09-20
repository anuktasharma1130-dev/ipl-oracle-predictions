"""
ml/test_prediction_engine.py
============================
Comprehensive test suite for the IPL Oracle ML Prediction Engine and Flask API.
Validates:
  1. Model loading from joblib files
  2. Pre-toss prediction generation
  3. Post-toss prediction generation
  4. Probability sum constraint (~1.0)
  5. Different teams validation (team1 != team2)
  6. Invalid team rejection
  7. Invalid toss rejection (invalid winner or decision)
  8. predict_proba() functionality on reloaded models
  9. API-compatible JSON schema and endpoint responses via Flask test client
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.app import app
from ml.prediction_engine import PredictionEngine, KNOWN_TEAMS


def test_model_loading(engine: PredictionEngine) -> None:
    """Test 1: Verify both pre-toss and post-toss models load properly."""
    print("Running Test 1: Model Loading...")
    assert engine.pre_toss_model is not None, "Pre-toss model failed to load."
    assert engine.post_toss_model is not None, "Post-toss model failed to load."
    assert len(engine.pre_toss_expected_features) == 36, (
        f"Expected 36 pre-toss features, got {len(engine.pre_toss_expected_features)}"
    )
    assert len(engine.post_toss_expected_features) == 40, (
        f"Expected 40 post-toss features, got {len(engine.post_toss_expected_features)}"
    )
    print(f"  [PASS] Pre-toss model loaded ({len(engine.pre_toss_expected_features)} features).")
    print(f"  [PASS] Post-toss model loaded ({len(engine.post_toss_expected_features)} features).")


def test_predict_proba_on_reload(engine: PredictionEngine) -> None:
    """Test 8: Verify predict_proba() executes on loaded pipeline."""
    print("Running Test 8: predict_proba() functionality on reloaded model...")
    df_pre = engine.build_feature_dataframe(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
        city="Mumbai",
    )
    probs_pre = engine.pre_toss_model.predict_proba(df_pre)[0]
    assert len(probs_pre) == 2, f"Expected 2 class probabilities, got {len(probs_pre)}"
    assert 0.0 <= probs_pre[0] <= 1.0, f"Probability out of bounds: {probs_pre[0]}"
    assert 0.0 <= probs_pre[1] <= 1.0, f"Probability out of bounds: {probs_pre[1]}"
    assert abs(sum(probs_pre) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {sum(probs_pre)}"

    df_post = engine.build_feature_dataframe(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        venue="Wankhede Stadium",
        city="Mumbai",
        toss_winner="Mumbai Indians",
        toss_decision="field",
    )
    probs_post = engine.post_toss_model.predict_proba(df_post)[0]
    assert len(probs_post) == 2, f"Expected 2 class probabilities, got {len(probs_post)}"
    assert 0.0 <= probs_post[0] <= 1.0, f"Probability out of bounds: {probs_post[0]}"
    assert 0.0 <= probs_post[1] <= 1.0, f"Probability out of bounds: {probs_post[1]}"
    assert abs(sum(probs_post) - 1.0) < 1e-4, f"Probabilities do not sum to 1.0: {sum(probs_post)}"
    print(f"  [PASS] Pre-toss predict_proba: {probs_pre}")
    print(f"  [PASS] Post-toss predict_proba: {probs_post}")


def test_pre_toss_prediction(engine: PredictionEngine) -> None:
    """Test 2: Verify pre-toss prediction structure and values."""
    print("Running Test 2: Pre-toss Prediction...")
    res = engine.predict(
        team1="Royal Challengers Bangalore",
        team2="Kolkata Knight Riders",
        venue="M Chinnaswamy Stadium",
    )
    assert res["model"] == "pre_toss", f"Expected model 'pre_toss', got {res['model']}"
    assert res["team1"] == "Royal Challengers Bangalore"
    assert res["team2"] == "Kolkata Knight Riders"
    assert "team1_win_probability" in res
    assert "team2_win_probability" in res
    assert "predicted_winner" in res
    assert res["predicted_winner"] in (res["team1"], res["team2"])
    assert res["confidence"] in ("Low", "Medium", "High")
    print(f"  [PASS] Pre-toss output verified for RCB vs KKR: Winner = {res['predicted_winner']} ({res['confidence']} conf).")


def test_post_toss_prediction(engine: PredictionEngine) -> None:
    """Test 3: Verify post-toss prediction structure and activation."""
    print("Running Test 3: Post-toss Prediction...")
    res = engine.predict(
        team1="Gujarat Titans",
        team2="Rajasthan Royals",
        venue="Narendra Modi Stadium",
        toss_winner="Gujarat Titans",
        toss_decision="field",
    )
    assert res["model"] == "post_toss", f"Expected model 'post_toss', got {res['model']}"
    assert res["team1"] == "Gujarat Titans"
    assert res["team2"] == "Rajasthan Royals"
    assert "team1_win_probability" in res
    assert "team2_win_probability" in res
    assert "predicted_winner" in res
    assert res["predicted_winner"] in (res["team1"], res["team2"])
    assert res["confidence"] in ("Low", "Medium", "High")
    assert "toss_winner" in res["details"]
    assert res["details"]["toss_winner"] == "Gujarat Titans"
    assert res["details"]["toss_decision"] == "field"
    print(f"  [PASS] Post-toss output verified for GT vs RR: Winner = {res['predicted_winner']}.")


def test_probability_sum(engine: PredictionEngine) -> None:
    """Test 4: Verify probability sum ~ 1.0 across diverse fixtures."""
    print("Running Test 4: Probability Sum Consistency...")
    test_cases = [
        ("Mumbai Indians", "Chennai Super Kings", "Wankhede Stadium", None, None),
        ("Delhi Capitals", "Punjab Kings", "Arun Jaitley Stadium", None, None),
        ("Sunrisers Hyderabad", "Lucknow Super Giants", "Rajiv Gandhi International Stadium", "Sunrisers Hyderabad", "bat"),
        ("Kolkata Knight Riders", "Mumbai Indians", "Eden Gardens", "Mumbai Indians", "field"),
    ]
    for t1, t2, v, tw, td in test_cases:
        res = engine.predict(team1=t1, team2=t2, venue=v, toss_winner=tw, toss_decision=td)
        p1 = res["team1_win_probability"]
        p2 = res["team2_win_probability"]
        total = p1 + p2
        assert abs(total - 1.0) < 1e-3, f"Probabilities {p1} + {p2} = {total} != 1.0 for {t1} vs {t2}"
    print(f"  [PASS] All {len(test_cases)} fixtures verified with sum(p) == 1.0.")


def test_different_teams_validation(engine: PredictionEngine) -> None:
    """Test 5: Verify rejection when team1 == team2."""
    print("Running Test 5: Different Teams Validation...")
    try:
        engine.predict(
            team1="Mumbai Indians",
            team2="Mumbai Indians",
            venue="Wankhede Stadium",
        )
        assert False, "Failed to raise ValueError for identical teams."
    except ValueError as e:
        assert "different" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected identical teams: {e}")

    # Also test with codes: MI vs Mumbai Indians
    try:
        engine.predict(
            team1="MI",
            team2="Mumbai Indians",
            venue="Wankhede Stadium",
        )
        assert False, "Failed to raise ValueError for alias match identical teams."
    except ValueError as e:
        assert "different" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected alias match identical teams: {e}")


def test_invalid_team_validation(engine: PredictionEngine) -> None:
    """Test 6: Verify rejection when an unknown team name is passed."""
    print("Running Test 6: Invalid Team Validation...")
    try:
        engine.predict(
            team1="Mars Invaders",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
        )
        assert False, "Failed to raise ValueError for unknown team1."
    except ValueError as e:
        assert "unknown team" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected unknown team1: {e}")

    try:
        engine.predict(
            team1="Chennai Super Kings",
            team2="Atlantis Warriors",
            venue="MA Chidambaram Stadium",
        )
        assert False, "Failed to raise ValueError for unknown team2."
    except ValueError as e:
        assert "unknown team" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected unknown team2: {e}")


def test_invalid_toss_validation(engine: PredictionEngine) -> None:
    """Test 7: Verify rejection for invalid toss parameters."""
    print("Running Test 7: Invalid Toss Validation...")
    # 1. Toss winner is neither team1 nor team2
    try:
        engine.predict(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            toss_winner="Kolkata Knight Riders",
            toss_decision="bat",
        )
        assert False, "Failed to raise ValueError for 3rd party toss winner."
    except ValueError as e:
        assert "must be either" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected invalid toss winner: {e}")

    # 2. Invalid toss decision (not 'bat' or 'field')
    try:
        engine.predict(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            toss_winner="Mumbai Indians",
            toss_decision="sleep",
        )
        assert False, "Failed to raise ValueError for invalid toss decision."
    except ValueError as e:
        assert "either 'bat' or 'field'" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected invalid toss decision: {e}")

    # 3. Only one toss parameter provided
    try:
        engine.predict(
            team1="Mumbai Indians",
            team2="Chennai Super Kings",
            venue="Wankhede Stadium",
            toss_winner="Mumbai Indians",
            toss_decision=None,
        )
        assert False, "Failed to raise ValueError for partial toss specification."
    except ValueError as e:
        assert "toss_decision must be specified" in str(e).lower(), f"Unexpected error message: {e}"
        print(f"  [PASS] Correctly rejected partial toss info: {e}")


def test_api_compatibility_and_endpoints() -> None:
    """Test 9: Verify Flask API endpoints via test client."""
    print("Running Test 9: Flask API Endpoints & JSON Schema Validation...")
    client = app.test_client()

    # GET /api/health
    resp_health = client.get("/api/health")
    assert resp_health.status_code == 200, f"Health endpoint failed: {resp_health.status_code}"
    health_data = resp_health.get_json()
    assert health_data["status"] == "ok"
    assert health_data["service"] == "IPL Oracle ML API"
    print(f"  [PASS] /api/health: {health_data}")

    # GET /api/model-info
    resp_info = client.get("/api/model-info")
    assert resp_info.status_code == 200, f"Model info endpoint failed: {resp_info.status_code}"
    info_data = resp_info.get_json()
    assert "available_models" in info_data
    assert "pre_toss" in info_data["available_models"]
    assert "post_toss" in info_data["available_models"]
    assert "evaluation_protocol" in info_data
    print(f"  [PASS] /api/model-info returned models: {list(info_data['available_models'].keys())}")

    # POST /api/predict (Pre-Toss valid)
    payload_pre = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
        "city": "Mumbai"
    }
    resp_pred_pre = client.post("/api/predict", json=payload_pre)
    assert resp_pred_pre.status_code == 200, f"Predict failed: {resp_pred_pre.status_code}, {resp_pred_pre.data}"
    pred_data_pre = resp_pred_pre.get_json()
    assert pred_data_pre["status"] == "success"
    assert pred_data_pre["model"] == "pre_toss"
    assert "team1_win_probability" in pred_data_pre
    assert "team2_win_probability" in pred_data_pre
    assert "predicted_winner" in pred_data_pre
    assert "confidence" in pred_data_pre
    print(f"  [PASS] POST /api/predict (Pre-Toss): {pred_data_pre['predicted_winner']} (p1={pred_data_pre['team1_win_probability']}, p2={pred_data_pre['team2_win_probability']})")

    # POST /api/predict (Post-Toss valid)
    payload_post = {
        "team1": "Mumbai Indians",
        "team2": "Chennai Super Kings",
        "venue": "Wankhede Stadium",
        "city": "Mumbai",
        "toss_winner": "Mumbai Indians",
        "toss_decision": "bat"
    }
    resp_pred_post = client.post("/api/predict", json=payload_post)
    assert resp_pred_post.status_code == 200, f"Predict post-toss failed: {resp_pred_post.status_code}"
    pred_data_post = resp_pred_post.get_json()
    assert pred_data_post["status"] == "success"
    assert pred_data_post["model"] == "post_toss"
    print(f"  [PASS] POST /api/predict (Post-Toss): {pred_data_post['predicted_winner']} (p1={pred_data_post['team1_win_probability']}, p2={pred_data_post['team2_win_probability']})")

    # POST /api/predict (Validation Error - same teams)
    resp_err = client.post("/api/predict", json={
        "team1": "Royal Challengers Bangalore",
        "team2": "Royal Challengers Bangalore",
        "venue": "M Chinnaswamy Stadium",
    })
    assert resp_err.status_code == 400, f"Expected 400 for identical teams, got {resp_err.status_code}"
    err_data = resp_err.get_json()
    assert err_data["status"] == "error"
    assert "different" in err_data["error"]
    print(f"  [PASS] POST /api/predict 400 Validation Check: {err_data['error']}")


def run_all_tests() -> None:
    """Execute complete test suite."""
    print("=" * 70)
    print("IPL ORACLE - INFERENCE ENGINE & API TEST SUITE")
    print("=" * 70)

    engine = PredictionEngine()

    test_model_loading(engine)
    test_predict_proba_on_reload(engine)
    test_pre_toss_prediction(engine)
    test_post_toss_prediction(engine)
    test_probability_sum(engine)
    test_different_teams_validation(engine)
    test_invalid_team_validation(engine)
    test_invalid_toss_validation(engine)
    test_api_compatibility_and_endpoints()

    print("=" * 70)
    print("ALL 9 TEST SUITES PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
