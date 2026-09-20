# IPL Oracle Machine Learning Pipeline Documentation

This document provides a detailed technical reference for the data processing, feature engineering, validation methodology, model development, and inference architecture of the IPL Oracle ML system.

---

## 1. Raw Data Extraction & Ingestion

### Source Data
- **Provider:** Cricsheet (Original CSV format).
- **Match Count:** 1,243 match files covering IPL tournaments from 2008 through 2026.
- **Format:** Hierarchical multi-record CSV files containing `version`, `info`, and `ball` rows.

### Ingestion Logic (`ml/parse_cricsheet.py`)
- Reads files directly without copying raw data into Git.
- Extracts match metadata from `info` rows: season, date, teams, toss winner, toss decision, venue, city, winner, eliminator, outcome, player of match.
- Normalizes franchise names via `TEAM_NAME_MAPPING`:
  - `Delhi Daredevils` $\rightarrow$ `Delhi Capitals`
  - `Kings XI Punjab` $\rightarrow$ `Punjab Kings`
  - `Deccan Chargers` $\rightarrow$ `Deccan Chargers` (historical franchise preserved)
  - `Rising Pune Supergiants` $\rightarrow$ `Rising Pune Supergiant`
  - `Royal Challengers Bangalore` $\rightarrow$ `Royal Challengers Bangalore`
- Resolves missing cities from stadium names via `VENUE_CITY_MAPPING` (e.g., Dubai International Stadium $\rightarrow$ Dubai, Sharjah Cricket Stadium $\rightarrow$ Sharjah).

### Decisive Match Filtering (`ml/build_dataset.py`)
- Filters out 9 abandoned/no-result fixtures.
- Retains 1,234 decisive matches.
- Constructs binary target `team1_win`:
  $$\text{team1\_win} = \begin{cases} 1 & \text{if winner} = \text{team1} \\ 0 & \text{if winner} = \text{team2} \end{cases}$$
- Resulting target distribution: 617 wins (50.0%) for Team 1 and 617 wins (50.0%) for Team 2.
- Exports base dataset to `data/processed/ipl_matches_base.csv`.

---

## 2. Chronological Feature Engineering

### Leakage-Prevention Paradigm
Every feature for match $T$ is calculated strictly using records from matches $1, \dots, T-1$. At no point does the current match's result, innings progression, player of the match, or toss outcome (for pre-toss models) enter the feature matrix.

### Feature Specification (`ml/feature_engineering.py`)

#### A. Cumulative Team Performance
- **Prior Win Rate:**
  $$\text{prior\_win\_rate}(T) = \frac{\sum_{i=1}^{T-1} \mathbb{I}(\text{won}_i)}{\text{matches\_played}}$$
- **Win Rate Differential:**
  $$\text{prior\_win\_rate\_diff} = \text{team1\_prior\_win\_rate} - \text{team2\_prior\_win\_rate}$$
- **Rolling Form:** Win rates over the last 3, 5, and 10 matches prior to match $T$.
- **Current Season Form:** Win rate strictly within the current tournament edition ($S$).

#### B. Dynamic Elo Rating System
- Initial rating: $R_0 = 1500.0$ for all franchises.
- Expected outcome:
  $$E_1 = \frac{1}{1 + 10^{(R_2 - R_1)/400}}, \quad E_2 = 1 - E_1$$
- Rating update ($K = 32$):
  $$R_1' = R_1 + K(S_1 - E_1)$$
- **Elo Momentum:** Delta between current Elo rating and rating 5 matches prior ($R(T) - R(T-5)$).
- **Logistic Elo Proxy:** $E_1$ provided as a continuous normalized probability feature (`elo_prob_t1`).

#### C. Matchup & Ground Familiarity
- **Head-to-Head Win Rate:** Historical win rate of Team 1 against Team 2.
- **Venue Win Rate:** Historical franchise win rate at the specific stadium.
- **Venue Chase Win Rate:** Historical percentage of matches won by chasing teams at the stadium:
  $$\text{venue\_chase\_win\_rate} = \frac{\text{chasing\_wins}}{\text{total\_venue\_matches}}$$
- **Home Ground Advantage:**
  $$\text{team1\_home\_advantage} = \mathbb{I}(\text{city} \in \text{Home}_1) - \mathbb{I}(\text{city} \in \text{Home}_2) \in \{-1, 0, 1\}$$

#### D. Tactical Toss Features (Post-Toss Model Only)
- `team1_toss_winner` ($\{0, 1\}$)
- `toss_decision_field` ($\{0, 1\}$)
- `team1_is_chasing` ($\{0, 1\}$)
- `team1_chase_venue_adv`: Interaction feature between chasing status and ground chase bias:
  $$\text{team1\_chase\_venue\_adv} = (2 \cdot \text{team1\_is\_chasing} - 1) \times (\text{venue\_chase\_win\_rate} - 0.50)$$

---

## 3. Evaluation & Validation Protocol

### Temporal Split
Sports data violates the i.i.d. assumption. Standard K-fold cross-validation results in temporal leakage (training on future matches to predict the past). We use a strictly chronological partition:
- **Training Set (2008–2023):** 1,019 matches
- **Validation Set (2024):** 71 matches (used for hyperparameter tuning and model selection)
- **Test Set (2025–2026):** 144 matches (held out untouched until final reporting)

---

## 4. Model Architectures & Selection

### Evaluated Model Classes
1. **Random Forest Classifier (Baseline):** 500 unconstrained trees. Suffered from severe over-fitting to nominal `season` indicators and high-cardinality one-hot encodings.
2. **Random Forest Classifier (Tuned):** Constrained depth (`max_depth=6`, `min_samples_leaf=8`), feature pruning.
3. **HistGradientBoostingClassifier:** Bins continuous features into 256 integer-valued bins, natively handles categorical interactions, applies L2 regularization, and employs early stopping on validation loss.
4. **Logistic Regression Benchmark:** Standard L2-regularized linear model.

### Comparative Results on Untouched 2025–2026 Test Set
- **Pre-Toss HistGradientBoosting:** Test Accuracy = 55.56%, Test ROC-AUC = 0.5403, Log Loss = 0.7013
- **Pre-Toss Random Forest Baseline:** Test Accuracy = 47.92%, Test ROC-AUC = 0.4693
- **Post-Toss HistGradientBoosting:** Test Accuracy = 50.69%, Test ROC-AUC = 0.4766
- **Post-Toss Tuned Random Forest:** Test Accuracy = 53.47%, Test ROC-AUC = 0.4760

---

## 5. Production Inference Pipeline

The inference layer (`ml/prediction_engine.py`) loads the trained scikit-learn pipelines directly via `joblib`:
1. Reconstructs historical accumulators from `ipl_matches_base.csv` up to the prediction horizon.
2. Normalizes user-supplied franchise names and stadium identifiers.
3. Selects the appropriate pipeline:
   - If toss information is absent: Routes to `pre_toss_improved_histgb.joblib` (36 features).
   - If toss information is present: Routes to `post_toss_improved_histgb.joblib` (40 features).
4. Invokes `pipeline.predict_proba()` to compute uncalibrated class likelihoods.
5. Verifies normalization ($P_1 + P_2 \approx 1.0$) and returns structured analytics.
