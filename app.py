"""Streamlit interface for the trained apartment price model."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.model import FEATURES, prepare_data, sha256

st.set_page_config(page_title='Morocco apartment prices', page_icon='🏠', layout='centered')
st.title('🏠 Morocco apartment price estimator')
st.write('Estimate an advertised apartment price from its location and characteristics.')
root = Path(__file__).resolve().parent
artifacts = root / 'artifacts'
try:
    run_id = json.loads((artifacts / 'latest.json').read_text())['run_id']
    run_dir = artifacts / 'runs' / run_id
    report = json.loads((run_dir / 'metrics.json').read_text())
    model = joblib.load(run_dir / 'model.joblib')
    df, _, _ = prepare_data(root / 'appartements_data.csv')
except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
    st.error(f'Could not load the trained model or dataset: {exc}. Run `python train.py` first.')
    st.stop()
if sha256(root / 'appartements_data.csv') != report['data_sha256']:
    st.warning('The dataset has changed since this model was trained. Run training again to update the model.')
cities = sorted(df['city_name'].dropna().unique().tolist())
with st.form('prediction'):
    city = st.selectbox('City', cities)
    area = st.number_input('Surface area (m²)', min_value=1.0, value=100.0, step=1.0)
    rooms = st.number_input('Bedrooms', min_value=0, value=2, step=1)
    baths = st.number_input('Bathrooms', min_value=0, value=1, step=1)
    salons = st.number_input('Living rooms', min_value=0, value=1, step=1)
    submitted = st.form_submit_button('Estimate price', type='primary')
if submitted:
    row = pd.DataFrame([{'city_name': city, 'surface_area': area, 'nb_rooms': rooms,
                         'nb_baths': baths, 'salon': salons}], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    st.metric('Estimated advertised price', f'{prediction:,.0f} MAD')
    st.caption('This is a statistical estimate, not a property valuation. Location within a city, condition, and other factors are not included.')
with st.expander('Model and evaluation'):
    st.write(f"Model: {report['selected_model']} · Run: {run_id}")
    st.write(f"Held-out MAE: {report['test_metrics']['mae_dh']:,.0f} MAD · R²: {report['test_metrics']['r2']:.3f}")
    st.caption('The held-out split groups rows by listing link. MAE is the average absolute error on this dataset; it is not an uncertainty interval.')
