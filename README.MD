# Retail Revenue Forecasting

<p align="center">
  <img src="https://img.shields.io/badge/Task-Retail%20Revenue%20Forecasting-01696f?style=for-the-badge" alt="Task badge" />
  <img src="https://img.shields.io/badge/Model-CatBoost-0c4e54?style=for-the-badge" alt="Model badge" />
  <img src="https://img.shields.io/badge/Validation-Rolling%20Origin-437a22?style=for-the-badge" alt="Validation badge" />
  <img src="https://img.shields.io/badge/Status-Portfolio%20Ready-da7101?style=for-the-badge" alt="Status badge" />
</p>

> Leakage-free multi-series retail revenue forecasting with gradient boosting, deep learning benchmarks, rolling validation, and final holdout evaluation.

## Executive Summary

This project builds an end-to-end forecasting system for retail revenue using daily aggregated time series across grouped product segments. The focus is not just model accuracy, but building a forecasting pipeline that is realistic, leakage-free, reproducible, and credible under future data conditions.

Unlike many portfolio forecasting projects that rely on random splits and overly optimistic metrics, this repository uses strict chronological validation, train-only preprocessing, feature ablation, rolling-origin tuning, and a final untouched test set. The result is a stronger demonstration of practical machine learning judgment, time-series reasoning, and production-style experimentation.

## Why This Project Stands Out

- **Business relevance:** Revenue forecasting supports planning, staffing, budgeting, promotions, and inventory strategy.
- **Technical rigor:** The pipeline avoids leakage through chronological splitting and train-only feature generation.
- **Fair experimentation:** Simple baselines, boosting models, and deep learning models are evaluated under one consistent framework.
- **Portfolio value:** The repository is organized like a real ML project, with modular code for data, features, training, evaluation, and utilities.

## Highlights

- Built a multi-series daily forecasting pipeline for grouped retail revenue.
- Engineered leakage-free calendar, lag, rolling, cyclical, and difference features.
- Ran feature ablation to identify which feature blocks actually improved forecasting quality.
- Used rolling-origin validation to estimate future performance more realistically.
- Compared classical baselines, gradient boosting, and deep learning benchmarks.
- Selected **CatBoost** as the strongest final model on the untouched test set.

## Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Problem Setup](#problem-setup)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Feature Ablation](#feature-ablation)
- [Models](#models)
- [Validation Strategy](#validation-strategy)
- [Evaluation](#evaluation)
- [Results Snapshot](#results-snapshot)
- [Forecast Plots](#forecast-plots)
- [How It Predicts](#how-it-predicts)
- [How to Run](#how-to-run)
- [Repository Layout](#repository-layout)
- [Roadmap](#roadmap)
- [Recruiter Notes](#recruiter-notes)
- [Contact](#contact)

## Overview

This project develops a research-grade forecasting pipeline for retail sales data using daily aggregated revenue series. The objective is to forecast future `Revenue` across grouped retail segments using leakage-free preprocessing, chronological validation, strong baseline models, and deep learning benchmarks.

The workflow is designed to be reproducible, modular, and easy to explain in academic, interview, and portfolio settings. It starts from raw transactional data and ends with final model comparison on an untouched test set.

## Business Problem

Retail teams need reliable forecasts to support inventory decisions, staffing, budgeting, and promotions. A forecasting model is only valuable if it performs well on truly unseen future data.

Many forecasting projects overstate performance by using random splits or preprocessing steps that leak future information. This project was intentionally designed to avoid those shortcuts and produce a more honest estimate of deployment-time performance.

## Dataset

The dataset contains retail transaction records with fields such as:

- Date
- Country
- State
- Product_Category
- Sub_Category
- Product
- Revenue

The final forecasting panel is built by aggregating daily `Revenue` at the grouped level:

- `Product_Category`
- `Sub_Category`

## Problem Setup

- **Target:** Revenue
- **Time frequency:** Daily
- **Forecast type:** Multi-series forecasting
- **Primary grouping level:** Product_Category + Sub_Category
- **Evaluation style:** Leakage-free, time-based forecasting evaluation

## Methodology

The project follows this end-to-end workflow:

1. Inspect and clean the raw transaction data.
2. Aggregate transactions into a daily grouped panel.
3. Build leakage-free time-series features.
4. Split the data chronologically into train, validation, and test sets.
5. Compare feature blocks using ablation.
6. Tune models using rolling-origin validation.
7. Retrain the best model on train + validation.
8. Evaluate once on the untouched test set.
9. Analyze performance at both overall and per-group levels.

## Feature Engineering

The final feature set includes:

- Calendar features such as day, week, month, quarter, year, and weekend indicators.
- Cyclical encodings for weekly and monthly seasonality.
- Lag features.
- Rolling statistics such as mean, std, min, and max.
- Difference features.
- Quarter and year boundary indicators.
- Outlier flag features.

All features are generated using past information only, which keeps the forecasting setup leakage-free and suitable for real future prediction.

## Feature Ablation

Before finalizing the pipeline, multiple feature combinations were tested to isolate what actually improved forecasting performance.

The experiments compared:

- Base features
- Base + calendar enhancements
- Base + difference features
- Base + outlier flags
- Base + all strong features together
- A winsorized version

The best result came from the full strong feature set. Winsorization did not improve performance and was removed from the final pipeline.

## Models

### Baselines

- Seasonal Naive
- Moving Average
- LightGBM
- CatBoost

### Deep Learning

- LSTM
- GRU
- Transformer Encoder

The best final model in the current pipeline was **CatBoost**.

## Validation Strategy

Time-series validation was used instead of random splitting.

The dataset was split chronologically into:

- 60% training
- 20% validation
- 20% test

For model tuning, rolling-origin validation was used so each candidate model was tested across multiple future windows rather than a single split. This gives a more realistic estimate of future performance and makes model selection more trustworthy.

## Evaluation

The project reports the following forecasting metrics:

- MAE
- RMSE
- SMAPE
- Median AE
- WAPE
- Per-group metrics

MAE and WAPE were the most useful metrics for comparing final forecasting quality because they capture both absolute error and scale-aware business relevance.

## Results Snapshot

| Model | MAE | RMSE | SMAPE | Median AE | WAPE | Approx. Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 241.01 | 905.76 | 44.00 | 102.75 | 5.01 | 94.99% |
| LightGBM | 244.42 | 1018.36 | 40.11 | 51.02 | 5.08 | 94.92% |
| Transformer | 2331.13 | 4966.12 | 99.77 | 836.61 | 47.37 | 52.63% |
| GRU | 2524.14 | 5452.90 | 78.72 | 937.30 | 51.30 | 48.70% |
| LSTM | 2537.83 | 5211.66 | 80.67 | 1095.15 | 51.57 | 48.43% |
| Moving Average | 2031.01 | 4400.53 | 67.57 | 648.64 | 42.19 | 57.81% |
| Seasonal Naive | 2648.10 | 5804.60 | 77.58 | 852.00 | 55.01 | 44.99% |

| Component | Outcome |
|---|---|
| Best final model | CatBoost |
| Benchmark baselines | Seasonal Naive, Moving Average |
| Validation approach | Rolling-origin time-series validation |
| Final test policy | Single untouched holdout evaluation |
| Best feature setup | Full strong feature set |
| Removed experiment | Winsorization |

CatBoost outperformed the simple moving-average and seasonal-naive baselines, showing that the engineered features added meaningful predictive signal.

## Forecast Plots

### Actual vs Predicted

![CatBoost Actual vs Pred](/outputs/final/figures/catboost_actual_vs_pred.png)

This plot shows how CatBoost predictions track actual revenue on the holdout data.

### Error Distribution

![CatBoost Error Distribution](/outputs/final/figures/catboost_error_distribution.png)

This plot helps assess whether the model errors are centered and whether large misses occur frequently.

### Group-wise MAE

![CatBoost Group MAE](/outputs/final/figures/catboost_group_mae.png)

This plot highlights which grouped retail segments are easier or harder to forecast.

## How It Predicts

The model predicts future `Revenue` for each `Product_Category + Sub_Category` group using only historical information.

For a future date, the pipeline:

1. Builds the same leakage-free features from past rows.
2. Applies the trained CatBoost model to the prepared feature matrix.
3. Produces one forecast per grouped row.
4. Compares those forecasts with actual revenue in the holdout period.

This means the model forecasts from structured time-series features, not raw transactions directly.

## How to Run

```bash
pip install -r requirements.txt
python -m scripts.train_baselines
python -m scripts.feature_ablation
python -m scripts.tune_rolling_models
python -m scripts.train_deep_models
python -m scripts.final_train_test
```

## Repository Layout

```text
retail-revenue-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml
├── configs/
│   ├── data_config.yaml
│   ├── feature_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── data/
│   ├── raw/
│   │   └── Sales.csv
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_build_panel.ipynb
│   ├── 03_train_baselines.ipynb
│   ├── 04_train_deep_models.ipynb
│   └── 05_error_analysis.ipynb
├── outputs/
│   ├── figures/
│   ├── final/
│   ├── logs/
│   ├── metrics/
│   └── models/
├── scripts/
│   ├── run_eda.py
│   ├── build_panel.py
│   ├── feature_ablation.py
│   ├── final_train_test.py
│   ├── train_baselines.py
│   ├── train_deep_models.py
│   ├── evaluate.py
│   ├── tune_rolling_models.py
│   └── run_all.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── clean_data.py
│   │   ├── build_panel.py
│   │   └── split_data.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── calendar_features.py
│   │   ├── lag_features.py
│   │   ├── rolling_features.py
│   │   ├── encoders.py
│   │   ├── scalers.py
│   │   ├── target_transform.py
│   │   ├── extra_features.py
│   │   └── outlier_treatment.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baselines.py
│   │   ├── lstm.py
│   │   ├── gru.py
│   │   └── transformer.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── datasets.py
│   │   ├── train_baselines.py
│   │   ├── rolling_validation.py
│   │   └── train_deep.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── group_metrics.py
│   │   ├── plots.py
│   │   └── error_analysis.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── logger.py
│       └── seed.py
└── tests/
```

## Roadmap

- [x] Data inspection
- [x] Build daily forecasting panel
- [x] Train leakage-free baselines
- [x] Add rolling validation
- [x] Add feature ablation
- [x] Add diagnostics and leaderboard
- [x] Add deep learning benchmarks
- [x] Polish repo for GitHub showcase

## Recruiter Notes

This project demonstrates the kinds of skills hiring teams look for in junior data scientists and ML engineers:

- Framing a forecasting problem clearly
- Building modular and reproducible ML pipelines
- Preventing data leakage
- Designing realistic validation strategies
- Comparing models fairly
- Translating experimentation into clear technical documentation

It shows more than model training. It shows practical ML judgment, reproducibility, and the ability to structure a project in a way that is easy for collaborators, reviewers, and recruiters to evaluate.

## Contact

- **Amit Vaghela**
- LinkedIn: [linkedin.com/in/amit-vaghela-436355102](https://www.linkedin.com/in/amit-vaghela-436355102)
- GitHub: [github.com/amitvaghela19](https://github.com/amitvaghela19)
- Email: amitvaghela19@gmail.com
- Location: Brampton, Ontario, Canada
