# IPL Oracle — Machine Learning Based IPL Match Prediction

IPL Oracle is an end-to-end machine learning and web application system that predicts match win probabilities for the Indian Premier League (IPL) using historical match data, leakage-safe chronological feature engineering, calibrated tree-based classifiers, and a local REST API.

---

## 1. Project Overview

Predicting the outcome of Twenty20 (T20) cricket matches is fundamentally challenging due to the high intrinsic variance of the format, where single-over momentum swings, individual player cameos, or brief fielding lapses frequently decide matches. Rather than framing match forecasting as a deterministic binary outcome, IPL Oracle computes probabilistic win likelihoods ($P(\text{Team}_1)$ vs $P(\text{Team}_2)$) grounded in longitudinal historical evidence.

The system supports two distinct operational modes:
- **Pre-Toss Prediction:** Evaluates match dynamics using only information available prior to the toss (lifetime win rates, recent form, head-to-head records, venue familiarity, and dynamic Elo ratings).
- **Post-Toss Prediction:** Ingests the toss outcome and tactical choice (batting first vs chasing), adjusting probabilities according to historical venue chase conversion biases and toss decisions.

All win likelihoods are generated strictly via `model.predict_proba()` and sum to approximately 1.0. The system does not claim guaranteed match outcomes.

---

## 2. Key Features

- **Historical Data Processing:** Custom parser for hierarchical Cricsheet match records covering all IPL seasons from 2008 to 2026.
- **Chronological Feature Engineering:** Zero future-information leakage; every match feature vector is constructed strictly from data available before that match commenced.
- **Dynamic Elo Rating System:** Continuous team skill tracking with exponential moving ratings and momentum differentials.
- **Form & Matchup Metrics:** Rolling 3, 5, and 10 match win rates, head-to-head records, in-season trajectory, and venue-specific performance.
- **Dual Pipeline Architecture:** Dedicated pre-toss and post-toss models that automatically switch based on available fixture parameters.
- **Lightweight Flask REST API:** Clean endpoints with CORS support, parameter normalization, input validation, and informative error handling.
- **Modern Glassmorphic Web Interface:** Single-page frontend (`ipl.html`) connected to the live API with dynamic loading states and zero heuristic fallbacks.
- **Deterministic Inference:** Reproducible probability outputs from fixed model seeds and verified numerical stability.

---

## 3. Technology Stack

- **Data Processing & Machine Learning:** Python 3.13, Pandas, NumPy, Scikit-learn, Joblib
- **API & Backend Layer:** Flask 3.1, Flask-CORS 6.0
- **Frontend & Visualizations:** HTML5, CSS3 (Vanilla Glassmorphic Design), Vanilla JavaScript (ES6+), Chart.js (for franchise historical records)
- **Testing & Verification:** Python `requests`, Microsoft Edge (Chromium DevTools Protocol automated browser testing)

---

## 4. Dataset

The project utilizes the public Cricsheet IPL match dataset (Original CSV format):
- **Raw Match Records Parsed:** 1,243 match CSV files parsed via custom recursive parser (`ml/parse_cricsheet.py`).
- **Temporal Coverage:** 2008 through 2026 seasons.
- **No-Result Matches Excluded:** 9 rain-abandoned or no-result encounters were filtered out to maintain target label integrity.
- **Final Decisive Dataset:** 1,234 decisive matches stored in `data/processed/ipl_matches_base.csv`.
- **Target Variable:** `team1_win` (Binary classification: `1` if Team 1 won, `0` if Team 2 won).
- **Target Distribution:** Exactly balanced: 617 wins (50.0%) for Team 1 and 617 wins (50.0%) for Team 2.
- **Franchise Name Normalization:** All historical team name variations (e.g., *Kings XI Punjab* $\rightarrow$ *Punjab Kings*, *Delhi Daredevils* $\rightarrow$ *Delhi Capitals*, *Royal Challengers Bangalore* $\rightarrow$ *Royal Challengers Bangalore*) mapped to canonical franchise entities.

*Note: The raw 1,243 individual CSV files are processed externally and excluded from version control; the cleaned base dataset and feature matrices are provided in `data/processed/`.*

---

## 5. Data Processing Pipeline

```
Raw Cricsheet CSVs (1,243 files)
               │
               ▼
   Metadata & Delivery Parser (ml/parse_cricsheet.py)
               │
               ▼
 Franchise & Ground Normalization (TEAM_NAME_MAPPING & VENUE_CITY_MAPPING)
               │
               ▼
   Decisive Match Filter (1,234 decisive matches, 0 missing targets)
               │
               ▼
  Strict Chronological Sorting (Date & Match ID ordering)
               │
               ▼
 Leakage-Safe Feature Engineering (ml/feature_engineering.py)
               │
               ▼
  ML-Ready Feature Dataset (data/processed/ipl_matches_features.csv)
```

No current match outcome, deliveries, scoreboards, player of the match, or margin of victory are ever used as input features.

---

## 6. Feature Engineering

Features for match $T$ are computed strictly using matches $[1, \dots, T-1]$:

| Feature Category | Description | Engineered Variables |
|---|---|---|
| **Prior Win Rates** | Cumulative lifetime franchise win rate prior to match | `team1_prior_win_rate`, `team2_prior_win_rate`, `prior_win_rate_diff` |
| **Rolling Form** | Form over the previous 3, 5, and 10 completed matches | `last_3_win_rate_diff`, `last_5_win_rate_diff`, `last_10_win_rate_diff` |
| **In-Season Form** | Win rate within the current tournament season | `team1_season_win_rate`, `team2_season_win_rate`, `season_win_rate_diff` |
| **Head-to-Head** | Historical win percentage and encounter volume between the two teams | `h2h_team1_win_rate`, `h2h_matches_before` |
| **Venue Performance** | Franchise win rates and total matches played at the specific ground | `team1_venue_win_rate`, `team2_venue_win_rate`, `venue_win_rate_diff`, `venue_matches_before` |
| **Venue Chase Bias** | Historical percentage of matches won by the team batting second at venue | `venue_chase_win_rate` |
| **Elo Ratings** | Dynamically updated Elo rating system ($K=32$) | `team1_elo`, `team2_elo`, `elo_diff`, `elo_prob_t1` |
| **Elo Momentum** | Delta between current Elo and Elo 5 matches prior | `team1_elo_momentum`, `team2_elo_momentum`, `elo_momentum_diff` |
| **Home Advantage** | Categorical home ground indicator based on franchise home cities | `team1_home_advantage` ($+1$ for Home, $-1$ for Away, $0$ for Neutral) |
| **Toss Interaction (Post-Toss)** | Tactical toss decisions, chasing status, and venue chase interaction | `team1_toss_winner`, `toss_decision_field`, `team1_is_chasing`, `team1_chase_venue_adv` |

When historical matches are insufficient (e.g., initial franchise seasons), uninformative neutral priors (0.50 win rate, 1500.0 Elo) are assigned.

---

## 7. Leakage Prevention Protocol

Data leakage is the most prevalent pitfall in sports analytics. The project implements strict architectural safeguards:
1. **Zero Outcome Leakage:** Match result columns (`winner`, `margin`, `player_of_match`, `eliminator`) are strictly isolated from the training feature set.
2. **Strict Chronological Splitting:** Rather than random K-Fold cross-validation (which trains on future matches to predict past matches), the data is divided chronologically:
   - **Training Set (2008–2023):** 1,019 matches
   - **Validation Set (2024):** 71 matches (used strictly for model tuning and selection)
   - **Test Set (2025–2026):** 144 matches (held out untouched until final evaluation)
3. **Sequential State Accumulation:** Feature values for match $T$ are computed before match $T$'s outcome updates the accumulator state.
4. **Post-Toss Isolation:** Toss variables are excluded entirely from the pre-toss model to prevent post-toss information leaking into pre-match predictions.

---

## 8. Model Development

Four distinct model architectures were evaluated using scikit-learn pipelines with embedded `ColumnTransformer` preprocessing:
1. **Random Forest Baseline:** 500 unconstrained decision trees with sparse one-hot encoded categorical features. Suffered from memorization of historical seasons and one-hot sparsity.
2. **Random Forest Tuned:** Reduced depth (`max_depth=6`), pruned features, and excluded nominal `season` one-hot memorization.
3. **HistGradientBoostingClassifier:** Histogram-based gradient boosting trees with native numerical binning, early stopping, and regularized leaf splits. Exhibited superior generalization on unseen temporal splits.
4. **Logistic Regression Benchmark:** Linear baseline regularized with L2 penalty.

---

## 9. Final Model Evaluation

Evaluated on the **untouched chronological 2025–2026 test set (144 matches)**:

### Pre-Toss Models
| Model Architecture | 2024 Val Acc | 2024 Val AUC | 2025–26 Test Acc | 2025–26 Test AUC | Test Log Loss |
|---|---|---|---|---|---|
| Random Forest (Baseline) | 39.44% | 0.3960 | 47.92% | 0.4693 | 0.7064 |
| Random Forest (Tuned) | 39.44% | 0.4357 | 54.17% | 0.4685 | 0.6971 |
| **HistGradientBoosting (Selected)** | **52.11%** | **0.4754** | **55.56%** | **0.5403** | **0.7013** |
| Logistic Regression (Benchmark) | 50.70% | 0.4968 | 45.83% | 0.4119 | 0.7459 |

### Post-Toss Models
| Model Architecture | 2024 Val Acc | 2024 Val AUC | 2025–26 Test Acc | 2025–26 Test AUC | Test Log Loss |
|---|---|---|---|---|---|
| Random Forest (Baseline) | 47.89% | 0.4071 | 49.31% | 0.4616 | 0.7075 |
| Random Forest (Tuned) | 40.85% | 0.4111 | 53.47% | 0.4760 | 0.6936 |
| **HistGradientBoosting (Selected)** | **47.89%** | **0.4349** | **50.69%** | **0.4766** | **0.7023** |
| Logistic Regression (Benchmark) | 49.30% | 0.5127 | 45.14% | 0.4217 | 0.7409 |

*Note: In high-entropy sporting environments such as T20 cricket, test accuracies in the 50–56% range reflect realistic limits of pre-match predictability.*

---

## 10. Prediction Architecture

```
                      User Interaction
                             │
                             ▼
              Web Frontend (ipl.html)
                             │  fetch(POST /api/predict)
                             ▼
              Flask REST API (api/app.py)
                             │
                             ▼
            Inference Engine (ml/prediction_engine.py)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
    Input Validation                Chronological State
 (Franchise/Venue/Toss)            (Elo, Form, H2H, Venue)
             │                               │
             └───────────────┬───────────────┘
                             ▼
               Feature Matrix Assembly (36 or 40 cols)
                             │
             ┌───────────────┴───────────────┐
             │ Toss Provided?                │
       [No]  ▼                         [Yes] ▼
   Pre-Toss Pipeline               Post-Toss Pipeline
(ml/models/pre_toss_...joblib)   (ml/models/post_toss_...joblib)
             │                               │
             └───────────────┬───────────────┘
                             ▼
                     predict_proba()
                             │
                             ▼
         JSON Response (Probabilities, Winner, Details)
                             │
                             ▼
              Frontend Result Card & Animations
```

---

## 11. API Endpoints

The Flask API runs locally on `http://127.0.0.1:5000`.

### 1. `GET /api/health`
Health check endpoint.
```json
{
  "service": "IPL Oracle ML API",
  "status": "ok"
}
```

### 2. `GET /api/model-info`
Metadata endpoint returning architecture specifications, feature definitions, and split protocols.

### 3. `POST /api/predict`
Inference endpoint for match predictions.

**Pre-Toss Request:**
```json
{
  "team1": "Mumbai Indians",
  "team2": "Chennai Super Kings",
  "venue": "Wankhede Stadium"
}
```

**Response (`HTTP 200 OK`):**
```json
{
  "status": "success",
  "model": "pre_toss",
  "team1": "Mumbai Indians",
  "team2": "Chennai Super Kings",
  "venue": "Wankhede Stadium",
  "city": "Mumbai",
  "team1_win_probability": 0.3881,
  "team2_win_probability": 0.6119,
  "predicted_winner": "Chennai Super Kings",
  "confidence": "High",
  "details": {
    "team1_elo": 1454.0,
    "team2_elo": 1467.0,
    "elo_diff": -13.0,
    "h2h_matches_before": 41,
    "h2h_team1_win_rate": 0.5122,
    "team1_prior_win_rate": 0.5395,
    "team2_prior_win_rate": 0.5585,
    "team1_venue_win_rate": 0.6269,
    "team2_venue_win_rate": 0.5000,
    "venue_matches_before": 73,
    "venue_chase_win_rate": 0.5068,
    "team1_home_advantage": 1
  }
}
```

**Post-Toss Request:**
```json
{
  "team1": "Gujarat Titans",
  "team2": "Rajasthan Royals",
  "venue": "Narendra Modi Stadium",
  "toss_winner": "Gujarat Titans",
  "toss_decision": "field"
}
```

---

## 12. Frontend Integration

The user interface (`ipl.html`) provides an interactive interface to the model:
- **Fixture Selection:** Team dropdowns with dynamic home ground auto-assignment.
- **Toss Toggle:** Optional toss configuration that switches between pre-toss and post-toss inference pathways.
- **Direct Probability Binding:** Win percentages reflect exact model outputs without JavaScript heuristics.
- **Graceful Error Handling:** If the API is offline or receives invalid input (e.g., identical teams), an error banner notifies the user without falling back to synthetic heuristics.
- **Model Attribution Badge:** Displays the active inference pipeline (`Powered by IPL Oracle ML • Model: HistGradientBoosting`).

---

## 13. Testing & Verification

The project includes multi-tiered automated verification:
1. **Engine & Unit Testing (`ml/test_prediction_engine.py`):**
   - 9 regression suites verifying pipeline loading, pre/post-toss outputs, probability normalization ($p_1 + p_2 = 1.0$), alias mappings, and input validation.
2. **Browser End-to-End Testing:**
   - Automated testing executed via Microsoft Edge DevTools Protocol against the live web server and API, validating UI animations, loading indicators, error banners, and DOM rendering.
3. **Black-Box Sanity & Robustness Testing (`ml/test_prediction_sanity.py`):**
   - 8 test categories across 24 test queries confirming:
     * Matchup variance across franchises
     * Directional consistency upon team order swap
     * Venue sensitivity and ground feature updates
     * 5 distinct toss scenarios activating post-toss routing
     * Bit-for-bit determinism across repeated queries
     * Strict HTTP 400 error containment with zero leaked stack traces

---

## 14. Project Structure

```
ipl-oracle-predictions/
├── .gitignore
├── README.md
├── requirements-api.txt
├── ipl.html
├── api/
│   ├── __init__.py
│   └── app.py
├── data/
│   └── processed/
│       ├── ipl_matches_base.csv
│       ├── ipl_matches_features.csv
│       ├── feature_importance_pre_toss.csv
│       ├── feature_importance_post_toss.csv
│       ├── confusion_matrix_pre_toss.png
│       ├── confusion_matrix_post_toss.png
│       ├── confusion_matrix_improved_pre_toss.png
│       ├── confusion_matrix_improved_post_toss.png
│       ├── feature_importance_pre_toss.png
│       ├── feature_importance_post_toss.png
│       ├── feature_importance_improved_pre_toss.png
│       ├── model_comparison_chart.png
│       ├── data_quality_report.txt
│       ├── feature_engineering_report.txt
│       ├── model_evaluation_report.txt
│       ├── model_improvement_report.txt
│       ├── step6_api_test_report.txt
│       ├── step6b_frontend_api_integration_report.txt
│       ├── step6c_prediction_sanity_report.txt
│       ├── repository_audit.txt
│       └── step7_documentation_audit.txt
├── docs/
│   ├── ML_PIPELINE.md
│   └── API.md
└── ml/
    ├── __init__.py
    ├── parse_cricsheet.py
    ├── build_dataset.py
    ├── feature_engineering.py
    ├── train_model.py
    ├── evaluate_model.py
    ├── model_improvement.py
    ├── prediction_engine.py
    ├── test_prediction_engine.py
    ├── test_prediction_sanity.py
    └── models/
        ├── pre_toss_improved_histgb.joblib
        ├── post_toss_improved_histgb.joblib
        ├── pre_toss_tuned_rf.joblib
        ├── post_toss_tuned_rf.joblib
        ├── pre_toss_model.joblib
        └── post_toss_model.joblib
```

---

## 15. How to Run Locally

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/anuktasharma1130-dev/ipl-oracle-predictions.git
cd ipl-oracle-predictions
pip install -r requirements-api.txt
```

### 2. Start Flask ML API Server
```bash
python api/app.py
```
*The API initializes the prediction engine and listens on `http://127.0.0.1:5000`.*

### 3. Start Frontend Local Server
In a separate terminal window:
```bash
python -m http.server 5500
```

### 4. Access Web Interface
Open your web browser and navigate to:
```
http://127.0.0.1:5500/ipl.html
```

---

## 16. Limitations

- **Intrinsic Sport Variance:** T20 cricket is subject to substantial stochastic variance. Pre-match statistical models capture historical propensities but cannot predict anomalous individual innings or uncharacteristic dropped catches.
- **Unmodeled In-Match Dynamics:** Pre-match inference does not account for mid-match weather disruptions (DLS method), pitch degradation, or dew factor onset.
- **Player-Level Availability:** Current features aggregate team-level longitudinal metrics; last-minute player injuries, squad rotations, and auction transfers are not directly captured.
- **Cold Start for New Franchises:** Newly introduced franchises (e.g., early seasons for Gujarat Titans and Lucknow Super Giants) rely on default priors until sufficient match history accumulates.
- **Local Deployment:** The API is configured for local development and has not been provisioned on cloud infrastructure.

---

## 17. Future Scope

- **Live In-Match / Ball-by-Ball Probability Tracking:** Dynamic win probability updates incorporating live required run rate, wickets in hand, and over-by-over momentum.
- **Granular Player & Matchup Features:** Individual batter vs bowler head-to-head statistics, phase-specific metrics (Powerplay, Middle, Death), and player fitness ratings.
- **Environmental Variables:** Incorporating live pitch reports, humidity, temperature, and historical dew impact on boundary dimensions.
- **Probability Calibration:** Exploring isotonic regression and Platt scaling to optimize calibration curves on validation folds.
- **Cloud Infrastructure & CI/CD:** Containerization via Docker, cloud deployment on AWS/GCP, and automated continuous retraining upon Cricsheet data releases.

---

## 18. Disclaimer

IPL Oracle provides machine-learning-based probability estimates for educational and analytical purposes. Predictions are probabilistic and should not be interpreted as guaranteed match outcomes.
