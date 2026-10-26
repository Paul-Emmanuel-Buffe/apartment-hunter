# Apartment Hunter ("Chasseurs d'appart'")

End-to-end machine learning project: it estimates the sale price of a house in **King County (Seattle area, USA)** from its characteristics. The project covers data cleaning, exploratory analysis in Power BI, model training and comparison, and deployment as a **Flask web app** packaged with **Docker**.

The model is a tuned Gradient Boosting regressor trained on King County sales from **May 2014 to May 2015**. The web interface is in French. The app offers both a web form and a JSON API.

## Table of contents

1. [Project pipeline](#project-pipeline)
2. [Data](#data)
3. [Modelling results](#modelling-results)
4. [Project structure](#project-structure)
5. [Run locally](#run-locally)
6. [Run with Docker](#run-with-docker)
7. [API](#api)
8. [Feature engineering at prediction time](#feature-engineering-at-prediction-time)
9. [Limitations](#limitations)

## Project pipeline

```
kc_house_data.csv (raw)
   │  Notebook 1: cleaning (duplicates, outliers)
   ▼
king_county_cleaned.csv ──► Power BI EDA (eda_pre_cleaned_data.pbix)
   │  Notebook 1: feature engineering + correlation matrix
   ▼
king_county_cleaned.csv (with log_price, age features, sale month)
   │  Notebook 2: split, encoding, Boruta, Ridge / Random Forest / Gradient Boosting
   ▼
model/gradient_boosting_model.joblib + model/zipcode_encoding.joblib
   │
   ▼
Flask app (app.py) ──► Docker image
```

### 1. Data exploration and cleaning (`1_data_exploration_and_cleaning.ipynb`)

- The raw dataset has 21,613 rows and 21 columns, with no missing values and no strictly identical rows.
- 177 houses were sold more than once: only the **most recent sale** of each house is kept, to avoid data leakage between train and test.
- Outliers removed:
  - living area above 10,000 sqft
  - lot above 750,000 sqft (about 7 ha)
  - the 33-bedroom data-entry error and the atypical 11-bedroom house
  - houses with 0 bedrooms and more than 1,500 sqft (inconsistent)
  - houses with 0 bathrooms
- Result: **21,408 rows** (205 removed, about 0.95 %).
- The cleaned file is exported and explored in Power BI, then the notebook adds features:
  - `log_price` (log of the target, to handle the right-skewed price distribution)
  - `anciennete` (age in 2015), `est_renove` (renovated flag) and `age_effectif` (age since the last renovation, or since construction if never renovated)
  - `mois_vente` (month of sale, for seasonality)
- A Pearson correlation matrix highlights the main drivers of price (`grade` and `sqft_living` at about 0.70, then `sqft_living15`, `sqft_above`, `bathrooms`, `lat`) and the collinearity between area-related variables.

### 2. Modelling and evaluation (`2_modeling_and_evaluation.ipynb`)

- **Stratified split 80 / 20** on 5 price quantiles (`qcut` on `log_price`): 17,126 training rows and 4,282 test rows, 20 features. Cheap, average and luxury homes are represented in both sets.
- **Target encoding of the zip code** (mean `log_price` per zip code), computed on the training set only. Unknown zip codes in the test set fall back to the training mean.
- **Boruta feature selection** (Random Forest, 50 iterations) confirmed 12 features and rejected 8 (`bedrooms`, `bathrooms`, `floors`, `condition`, `sqft_basement`, `est_renove`, `age_effectif`, `mois_vente`), mostly because they are collinear with `sqft_living`. Boruta relies on shallow trees and underrates some variables that matter in the final model (`condition`, `age_effectif`), so **all 20 features were kept** and left to the models (and Ridge regularisation) to handle.
- Three algorithms were compared, then the best one was tuned with `GridSearchCV`.
- **Permutation importance** was computed on the test set to check that the model relies on sensible criteria.
- The final model and the zip code encoding table are saved with Joblib.

## Data

| File | Description |
|---|---|
| `data/kc_house_data.csv` | Raw King County house sales (21,613 rows, 21 columns) |
| `data/king_county_cleaned.csv` | Cleaned dataset with engineered features (21,408 rows), used for modelling |
| `data/zipcodes.json` | Per zip code: centre coordinates, median neighbour living and lot areas, number of sales. Used to pre-fill optional fields in the app |
| `powerbi/eda_pre_cleaned_data.pbix` | Power BI report for the exploratory analysis of the cleaned data |

The raw dataset is the well-known "King County House Sales" dataset (sales between May 2014 and May 2015).

## Modelling results

Metrics on the held-out test set. R² is computed on `log(price)`; MAE and RMSE are in US dollars after converting back with `exp`.

| Model | R² (log) | MAE | RMSE |
|---|---|---|---|
| Ridge (RobustScaler + RidgeCV, alpha = 1.87) | 0.8607 | 83,428 $ | 163,671 $ |
| Random Forest (100 trees, depth 15) | 0.8935 | 68,559 $ | 126,865 $ |
| HistGradientBoosting (default-ish) | 0.9071 | 64,383 $ | 123,200 $ |
| **HistGradientBoosting (tuned, champion)** | **0.9085** | **62,972 $** | **119,209 $** |

Best hyperparameters (24 combinations, 3-fold cross-validation): `learning_rate=0.05`, `max_iter=250`, `max_depth=None`, `min_samples_leaf=50`.

The app displays a median error of **8.1 %** on unseen houses and uses it to build the indicative price range.

Top 10 variables by permutation importance:

| Variable | Importance |
|---|---|
| `zipcode_encoded` | 0.231 |
| `sqft_living` | 0.196 |
| `lat` | 0.159 |
| `grade` | 0.094 |
| `long` | 0.052 |
| `sqft_lot` | 0.033 |
| `view` | 0.017 |
| `age_effectif` | 0.013 |
| `sqft_living15` | 0.011 |
| `condition` | 0.009 |

Location (zip code, latitude, longitude) is the dominant factor, followed by living area and construction quality.

## Features of the web app

- Web form to describe a property (location, surfaces, rooms, age, quality, view, etc.)
- Price estimate with an indicative range based on the model's median error
- Price per square metre
- "What the model received" panel showing the exact features sent to the model
- JSON API and health check
- Input validation with clear error messages
- Optional advanced fields (latitude, longitude, neighbourhood averages) that default to values computed from the selected zip code

## Project structure

```
.
├── app.py                  # Flask application
├── requirements.txt
├── Dockerfile
├── README.md
├── data/
│   ├── kc_house_data.csv
│   ├── king_county_cleaned.csv
│   └── zipcodes.json
├── model/
│   ├── gradient_boosting_model.joblib   # trained model
│   └── zipcode_encoding.joblib          # zip code -> target encoding
├── notebooks/
│   ├── 1_data_exploration_and_cleaning.ipynb
│   └── 2_modeling_and_evaluation.ipynb
├── powerbi/
│   └── eda_pre_cleaned_data.pbix
├── templates/
│   └── index.html
└── static/
    └── style.css
```

`app.py` checks at startup that the zip codes in `zipcode_encoding.joblib` and `zipcodes.json` match, and stops with an error if they differ. The notebooks read and write files through relative paths (`../data/`, `../model/`), so they are meant to be run from the `notebooks/` folder.

## Requirements

- Python 3.10+ (the Docker image uses Python 3.11)
- Dependencies pinned in `requirements.txt`: Flask, scikit-learn, pandas, numpy, joblib
- To re-run the notebooks you also need `boruta`, `matplotlib` and `seaborn`, plus Jupyter
- Power BI Desktop to open the `.pbix` file

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The app starts on http://127.0.0.1:5001 by default.

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Interface to listen on |
| `PORT` | `5001` | Port to listen on |
| `FLASK_DEBUG` | unset | Set to `1` for debug mode with auto-reload |

## Run with Docker

Build the image:

```bash
docker build -t apartment-hunter .
```

Run it:

```bash
docker run --rm -p 5000:5000 apartment-hunter
```

Then open http://localhost:5000.

Useful variations:

```bash
# run in the background
docker run -d --name apartment-hunter -p 5000:5000 apartment-hunter

# use another port on your machine
docker run --rm -p 8080:5000 apartment-hunter

# logs, stop, remove
docker logs -f apartment-hunter
docker stop apartment-hunter
docker rm -f apartment-hunter
```

The Dockerfile sets `HOST=0.0.0.0` and `PORT=5000` so the app is reachable from outside the container.

## API

### `GET /sante`

Health check.

```json
{"statut": "ok"}
```

### `POST /api/estimation`

Send a JSON object. Surfaces are in square metres and are converted to square feet internally.

```bash
curl -X POST http://localhost:5000/api/estimation \
  -H "Content-Type: application/json" \
  -d '{
    "zipcode": "98118",
    "living_m2": 150,
    "basement_m2": 0,
    "lot_m2": 600,
    "bedrooms": 3,
    "bathrooms": 2,
    "floors": 1,
    "yr_built": 1975,
    "mois_vente": 5,
    "grade": 7,
    "condition": 3,
    "view": 0
  }'
```

The response contains `prix_estime`, `fourchette` (low and high bounds), `prix_m2`, `devise` (`USD`), `erreur_mediane_pct` and `variables_modele` (the features sent to the model).

On invalid input, the API returns HTTP 400 with an `erreurs` object listing the problems per field.

### Input fields

| Field | Description | Required | Range |
|---|---|---|---|
| `zipcode` | King County zip code (70 supported) | yes | see `data/zipcodes.json` |
| `living_m2` | Living area (m²) | yes | 30 - 930 |
| `above_m2` | Area above ground (m²); deduced if empty | no | 20 - 930 |
| `basement_m2` | Basement area (m²), 0 if none | yes | 0 - 900 |
| `lot_m2` | Lot size (m²) | yes | 40 - 70000 |
| `bedrooms` | Bedrooms (0 = studio) | yes | 0 - 10 |
| `bathrooms` | Bathrooms (0.25 steps) | yes | 0.5 - 8 |
| `floors` | Floors (0.5 steps) | yes | 1 - 3.5 |
| `yr_built` | Year built | yes | 1900 - 2015 |
| `yr_renovated` | Year of last renovation | no | 1900 - 2015 |
| `mois_vente` | Month of sale (1-12) | yes | 1 - 12 |
| `grade` | Construction quality | yes | 3 - 13 |
| `condition` | Overall condition | yes | 1 - 5 |
| `view` | View quality index | yes | 0 - 4 |
| `waterfront` | Waterfront property | no | true / false |
| `lat`, `long` | Coordinates; default to the zip code centre | no | 47.1 - 47.8 / -122.6 - -121.2 |
| `living15_m2`, `lot15_m2` | Average living / lot area of the 15 nearest neighbours; default to the zip code median | no | 30 - 930 / 40 - 70000 |

Validation rules: the basement must be smaller than the living area, `above_m2 + basement_m2` must match `living_m2` (within 1 m²), and the renovation year cannot precede the construction year.

## Feature engineering at prediction time

Before calling the model, `app.py` reproduces the same transformations as the notebooks:

- converts m² to ft² (1 m² = 10.7639 ft²)
- computes `anciennete`, `est_renove` and `age_effectif` (2015 is the reference year)
- fills in missing location and neighbourhood values from the zip code
- applies the zip code target encoding
- orders the columns exactly as the model expects (`feature_names_in_`)
- converts the prediction back from log scale with `exp`

## Limitations

- Trained on 2014-2015 data only: prices do not reflect today's market.
- Only valid for the King County zip codes present in `data/zipcodes.json`.
- The displayed range is the model's median error, not a statistical confidence interval.
- The app uses Flask's built-in development server. For real production use, run it behind a production WSGI server such as Gunicorn.
