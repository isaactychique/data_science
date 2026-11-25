# Morocco Apartment Price Prediction

A small end-to-end machine learning project for estimating advertised apartment prices in Morocco. It includes the original exploratory notebook, a reproducible training pipeline, local experiment tracking, and a Streamlit application.

## Setup

Use Python 3.10 from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and run

```bash
python train.py
python -m streamlit run app.py
```

The app requires a trained model. Training writes a new directory under `artifacts/runs/` for each run, containing `model.joblib` and `metrics.json`; `artifacts/latest.json` points to the latest completed run. These generated files are ignored by Git. The JSON report records the data hash, seed, split, candidate scores, final metrics, and library versions. Pass `--data`, `--output`, or `--seed` to `train.py` to change the defaults. The app reads the default dataset and artifacts directory.

Run checks with:

```bash
python -m unittest discover -s tests -v
python -m pip check
```

## Method and limitations

Rows without a valid positive price are excluded. The model uses city, surface area, bedroom count, bathroom count and living-room count. Missing numeric values are imputed using training data, and city names are one-hot encoded with unseen categories supported. Rows sharing an exact listing URL are kept in the same split. Three-fold grouped cross-validation on the training set selects between random forest and histogram gradient boosting by MAE. A separate grouped holdout is evaluated once, alongside a median-price baseline. No outlier filtering is applied; extreme values can strongly affect the scores. URL grouping does not detect the same listing under different URLs.

The app reports the average absolute error on the holdout as context, not a prediction interval. Prices are asking prices from this historical dataset, not necessarily current market prices or transaction prices. The dataset has missing and potentially erroneous values, limited geographic detail, and does not establish a representative sample of Morocco. Estimates should not be treated as valuations.

The experiment tracking here is local JSON and model files, not a deployed tracking server or production monitoring system. `joblib` artifacts should only be loaded from trusted sources and require compatible library versions. Direct dependency versions are pinned, but transitive dependencies are not locked.

## Original notebook

```bash
python -m ipykernel install --prefix "$VIRTUAL_ENV" --name appartements --display-name "Python (appartements)"
python -m jupyter lab
```

Open `appartements-price-prediction.ipynb` and choose **Python (appartements)**. Keep `appartements_data.csv` in the notebook's working directory. The notebook's TensorFlow installation cell is redundant after setup. Its historical analysis predates the training pipeline and its scores should not be compared directly: it preprocesses data before splitting and uses different filtering and features.
