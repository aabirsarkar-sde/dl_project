# Remaining Useful Life Prediction for Industrial Engines

Predictive maintenance on the NASA C-MAPSS turbofan dataset (subset FD001).
**Phase 1:** engineered degradation features + classical ML (`notebooks/rul-prediction-t017-phase-1.ipynb`).
**Phase 2:** LSTM / temporal CNN on raw sensor windows, plus a hybrid LSTM + engineered features, on the same engine-level split (`notebooks/02_phase2_deep_learning.ipynb`).

## Problem
Given the sensor history of an engine up to cycle *t*, predict its Remaining Useful Life (RUL): the number of operating cycles left before failure. This is a regression task.

## Dataset
NASA C-MAPSS FD001: 100 training engines (run to failure), 100 test engines (cut off early), 21 sensors, 3 operating settings, one operating condition and one fault mode.
Source: NASA Prognostics Data Repository, "Turbofan Engine Degradation Simulation Data Set" (Saxena and Goebel, 2008).
The three FD001 files are in `data/`.

## Method
1. **Data checks and EDA:** missing values, duplicates, cycle continuity, lifetime and RUL distributions, sensor trends.
2. **Target:** RUL = last cycle − current cycle for each training engine (uncapped for the main model).
3. **Sensor selection:** drop sensors with training std < 0.01 (s1, s5, s6, s10, s16, s18, s19), leaving 14.
4. **Features (57):** cycle, 14 raw sensors, and per sensor a rolling mean, rolling slope (window 5) and health (drift from the engine's own first 10 cycles). All features are causal.
5. **Evaluation:** 5-fold `GroupKFold` on engine ID over all 100 training engines (mean ± std). Final models are retrained on all training engines and scored once on the 100 test engines at their last cycle.
6. **Models:** naive mean baseline, Elastic Net, Random Forest.
7. **Metrics:** RMSE and MAE, plus MAE for early (RUL ≥ 50) and late (RUL < 50) stages.
8. **Secondary comparison:** RUL capped at 125, scored on uncapped RUL.

## Results
Random Forest is best on every metric. CV = 5-fold GroupKFold mean over the 100 training engines; Test = last cycle of each of the 100 test engines. Early = true RUL at least 50.

| Model | Data | RMSE | MAE | MAE early | MAE late |
|---|---|---|---|---|---|
| Naive (mean RUL) | CV mean | 68.87 | 56.86 | 48.40 | 83.31 |
| Elastic Net | CV mean | 40.04 | 30.63 | 32.96 | 23.35 |
| Random Forest | CV mean | 34.78 | 22.79 | 28.47 | 5.04 |
| Naive (mean RUL) | Test | 52.62 | 40.88 | 21.44 | 86.24 |
| Elastic Net | Test | 27.98 | 23.68 | 25.02 | 20.55 |
| Random Forest | Test | 23.34 | 15.58 | 20.54 | 4.02 |

CV and test scores are not directly comparable: CV scores every row of held-out engines across their whole lifetime, while the test score uses one cycle per engine. Capping RUL at 125 scored worse on uncapped RUL (CV RMSE 46.40 vs 34.78), so the main model is uncapped.

## Phase 2 results (sequence models)
Same target, sensors, folds, metrics and single test evaluation as Phase 1. Input: 30-cycle windows of the 14 raw sensors (z-scored per fold). The Random Forest is re-run on the same folds (scores differ slightly from Phase 1 because of the scikit-learn version). Deep models on test: mean over 3 seeds.

| Model | CV RMSE | CV MAE | CV MAE late | Test RMSE | Test MAE |
|---|---|---|---|---|---|
| Random Forest (Phase 1 recipe) | 35.17 ± 1.52 | 23.03 | 5.02 | 23.32 | 15.57 |
| LSTM, raw windows (main) | 41.76 ± 1.70 | 29.81 | 12.33 | 25.33 ± 3.55 | 18.29 |
| Temporal CNN | 44.06 ± 3.56 | 31.69 | 15.19 | 26.27 ± 0.74 | 19.89 |
| MLP on 57 features (control) | 34.20 ± 3.27 | 22.03 | 5.23 | 21.59 ± 0.44 | 14.91 |
| Hybrid LSTM + features | 35.05 ± 2.19 | 23.76 | 6.00 | 25.23 ± 0.15 | 17.68 |

Ablations (CV RMSE): window 10 / 30 / 50 = 43.12 / 41.76 / 41.61; no dropout 40.89; hybrid capped at 125 = 46.02 on uncapped RUL.

**Finding:** raw 30-cycle windows alone do worse than engineered features. Adding the features closes the gap, but an MLP on the features does as well, so long-term context (cycle number, drift from the engine's healthy start) matters more than the model type on FD001.

## How to run
```bash
git clone <repo-url>
cd rul-prediction-cmapss
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in data/ (see data/README.md)
jupyter notebook notebooks/02_phase2_deep_learning.ipynb
```
The Phase 2 notebook finds the data in `../data/`, `data/` or the current folder, so it also runs in Colab after uploading the three files (GPU optional; about 25 minutes on a laptop CPU). The Phase 1 notebook reads the files from its working directory.

Rebuilding the Phase 2 report and slides:
```bash
python reports/scripts/extract_phase2_results.py   # numbers + figures from the executed notebook
python reports/scripts/phase2_diagrams.py          # window / architecture diagrams
node reports/scripts/build_phase2_deck.js          # needs: npm install pptxgenjs
cd reports && tectonic phase2_report.tex           # or upload to Overleaf with figures/
```


## Limitations
Fixed model settings (no tuning) in both phases; window sizes chosen on the same engines used for model comparison; Phase 2 ablations use one seed; baseline length and the late threshold set by hand; test score is one cycle per engine; FD001 only; uncapped target, so scores are not comparable to papers that cap RUL at 125.

## Team
Kush Puri 
Aadit Vachher 
Vachan Gupta 
Aabir Sarkar
Kushagra Maheshwari


## References
- Saxena, A., and Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. NASA Prognostics Data Repository.
- Hochreiter, S., and Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
- Zheng, S., Ristovski, K., Farahat, A., and Gupta, C. (2017). Long short-term memory network for remaining useful life estimation. IEEE ICPHM.
- Babu, G. S., Zhao, P., and Li, X.-L. (2016). Deep convolutional neural network based regression approach for estimation of remaining useful life. DASFAA.
- Srivastava, N., et al. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR.
- Breiman, L. (2001). Random Forests. Machine Learning.
- Zou, H., and Hastie, T. (2005). Regularization and variable selection via the elastic net. JRSS B.
