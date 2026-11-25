"""Shared data preparation and model definitions for apartment price prediction."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ['surface_area', 'nb_rooms', 'nb_baths', 'salon']
CATEGORICAL = ['city_name']
FEATURES = CATEGORICAL + NUMERIC
CITY_MAPPING = {'الدار البيضاء': 'Casablanca', 'الرباط': 'Rabat', 'مراكش': 'Marrakech',
                'أكادير': 'Agadir', 'القنيطرة': 'Kénitra', 'طنجة': 'Tanger', 'المحمدية': 'Mohammedia'}


def prepare_data(path: str | Path):
    df = pd.read_csv(path, na_values=['None'])
    required = set(FEATURES + ['price', 'link'])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    price = pd.to_numeric(df['price'].astype('string').str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    valid = price.notna() & np.isfinite(price) & (price > 0)
    dropped = int((~valid).sum())
    df = df.loc[valid].copy()
    df['price'] = price.loc[valid]
    df['city_name'] = df['city_name'].replace(CITY_MAPPING).str.strip()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[(df[col] < 0) | ~np.isfinite(df[col]), col] = np.nan
    # Missing links are distinct records, not one shared group.
    links = df['link'].fillna('').astype(str).str.strip()
    groups = links.where(links.ne(''), '__missing_link_' + df.index.astype(str))
    return df, groups, dropped


def make_model(name: str, seed: int = 42):
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), NUMERIC),
        ('city', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL),
    ])
    models = {
        'random_forest': RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=-1),
        'hist_gradient_boosting': HistGradientBoostingRegressor(max_iter=150, l2_regularization=1, random_state=seed),
    }
    if name not in models:
        raise ValueError(f'Unknown model: {name}')
    return Pipeline([('preprocess', preprocess), ('model', models[name])])


def sha256(path: str | Path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
