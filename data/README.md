# Explainable AI on Ames Housing — MICE-XGBoost Imputation & Tri-Model Comparison

An applied explainable AI (XAI) pipeline built on the Ames Housing dataset, comparing
Random Forest, Linear Regression, and XGBoost through the lens of `dalex` — with a
statistically grounded missing-data strategy underneath it.

## Motivation

Most XAI walkthroughs start from clean, complete toy datasets. This project instead
starts from a realistic, messy dataset (2,900+ rows, 79 features, ~25 columns with
missing values) and treats the missing-data problem as seriously as the modeling
problem — because feeding a model garbage imputations produces garbage explanations,
no matter how good your XAI tooling is.

## Pipeline overview

**1. Structural vs. MAR missingness split**
Before imputing anything, missing values were split into two categories:
- **Structural NaNs** — columns where `NaN` doesn't mean "missing," it means "this
  house doesn't have this feature" (e.g. `Pool QC` NaN = no pool, `Bsmt Qual` NaN =
  no basement). These were filled with an explicit `"None"` category, cross-checked
  against related numeric columns (e.g. confirming `Pool QC` is NaN wherever
  `Pool Area == 0`).
- **Genuinely missing (MAR-plausible)** — a much smaller set of columns
  (`Lot Frontage`, `Electrical`, a handful of `Bsmt`/`Garage` fields) where no such
  structural explanation applies. These were treated as Missing at Random and passed
  to the imputation stage.

**2. MICE-XGBoost imputation**
Multiple Imputation by Chained Equations, implemented manually with `XGBRegressor`
for numeric columns and `XGBClassifier` for categorical columns (rather than a single
shared estimator), since sklearn's `IterativeImputer` doesn't support mixed
estimator types natively. 5 imputed datasets were generated and pooled (mean for
numeric, mode for categorical) into a single working dataset.

**3. Tri-model comparison**
Three models were trained on the same target (`log(SalePrice)`, given the target's
right skew) for a fair comparison of very different inductive biases:
- **Linear Regression** — sensitive to multicollinearity; required dropping
  `Total Bsmt SF` (an exact linear combination of three other basement columns)
  to avoid numerically unstable coefficients.
- **Random Forest**
- **XGBoost**

**4. Explainability with `dalex`**
- **Permutation-based variable importance** (`model_parts`) — comparing which
  features each model relies on most.
- **Partial Dependence Profiles** (`model_profile`) — comparing how predictions
  change across the range of a feature. Notably, Linear Regression extrapolates
  linearly without limit, while Random Forest and XGBoost plateau in
  sparsely-populated regions of the feature space (e.g. very large houses),
  highlighting a real difference in how these models handle extrapolation.
- **SHAP values** (`predict_parts`, type `"shap"`) — local explanations for
  individual house price predictions, decomposing each prediction into
  per-feature contributions.

## Key findings

- Structural-NaN handling materially changes the missing-data picture: without it,
  columns like `Pool QC` (99%+ "missing") would have been misclassified as a MAR
  imputation target rather than correctly filled with `"None"`.
- Linear Regression's coefficients become numerically meaningless
  (`~1e55` magnitude) under exact multicollinearity — a good illustration of why
  linear models need explicit multicollinearity checks that tree-based models don't.
- Random Forest and XGBoost produce near-identical PDP shapes for continuous features
  like `Gr Liv Area`, both plateauing past ~3,000 sq ft where training data is sparse,
  while Linear Regression continues extrapolating linearly — a clear example of model
  behavior diverging outside the dense region of the data.

## Repo structure

```
xai-ames-housing/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ames_xai_pipeline.ipynb
└── data/
    └── README.md   (source + download instructions — raw data not committed)
```

## Data

Ames Housing dataset (Kaggle: `shashanknecrothapa/ames-housing-dataset`, originally
compiled by Dean De Cock). Not committed to this repo — download via:

```python
import kagglehub
path = kagglehub.dataset_download("shashanknecrothapa/ames-housing-dataset")
```

## Setup

```bash
pip install -r requirements.txt
```

## Tools

- `pandas`, `numpy` — data handling
- `scikit-learn` — Random Forest, Linear Regression, train/test splitting
- `xgboost` — XGBoost regressor/classifier, used both as the final model and as the
  imputation engine
- `dalex` — model-agnostic explainability (variable importance, PDP, SHAP)
