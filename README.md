# Morocco Apartment Price Prediction

An end-to-end learning project for estimating advertised apartment prices in Morocco: an exploratory notebook, a data audit, a training pipeline, local experiment tracking, and a Streamlit app.

## Setup and run

Python 3.10 is used for the tested environment. For the app and training only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python audit.py
python train.py
python -m streamlit run app.py
```

For the original notebook too, install `requirements.txt` instead. `requirements-lock.txt` records the complete environment used locally (Python 3.10/Linux); it can be installed with `pip install -r requirements-lock.txt` on a compatible platform. The app and Docker requirements pin direct dependencies, not all transitive packages.

The app uses `appartements_data.csv` and the latest local model. `python train.py --help` describes the data, output, and seed options. Each run is stored in `artifacts/runs/<run_id>/` with a model, a JSON report and `test_errors.csv` (held-out predictions and listing links). `artifacts/latest.json` points to the latest completed run. `python audit.py` writes `artifacts/data_audit.json`; its thresholds flag records for review and do not remove them. Artifacts are ignored by Git; do not load joblib models from untrusted sources.

```bash
python -m unittest discover -s tests -v
python -m pip check
```

GitHub Actions runs these checks on pushes and pull requests. For a local container, the following builds an image and trains a model during the build:

```bash
docker build -t morocco-apartment-prices .
docker run --rm -p 8501:8501 morocco-apartment-prices
```

The image is self-contained; its model is baked in, not synchronized with local `artifacts`. Rebuild it after changing data or training code. Local JSON tracking is not a deployed tracking server, registry, or production monitoring system.

## Evaluation and limitations

Rows without a valid positive price are excluded; suspicious extreme values are flagged but retained. The features are city, surface area, bedroom/bathroom/living-room counts, and optionally equipment. An empty equipment field is treated as unknown. Missing values are imputed inside each training fold, and city one-hot encoding supports unseen categories. Rows sharing an exact URL stay together, but the same listing under different URLs can still cross splits.

Three grouped folds on the training set compare random forest and histogram gradient boosting, with and without equipment and with and without a log-transformed target. Selection uses mean fold MAE. A grouped holdout is evaluated against a median baseline, with errors broken down by city and price band. Fold scores describe variation across folds, not full stability across random splits. Because the same holdout has already been inspected during development, it is **exploratory**; a fresh untouched dataset is needed for an independent final estimate. No calibrated prediction intervals are provided.

The data consists of historical asking prices, not necessarily current market or transaction prices. Some records contain missing or suspicious values, the sample may not represent Morocco, and features omit neighborhood and condition. Reported MAE is an aggregate, not a per-property uncertainty guarantee. Estimates are not valuations.

## Original notebook

Install `requirements.txt`, then run:

```bash
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name appartements --display-name "Python (appartements)"
python -m jupyter lab
```

Open `appartements-price-prediction.ipynb` and select **Python (appartements)**. Its TensorFlow installation cell is redundant after setup. The notebook predates the training pipeline, preprocesses before splitting, and uses different filtering and features; its scores should not be compared directly with the newer pipeline.
