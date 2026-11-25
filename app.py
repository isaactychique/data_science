"""Streamlit interface for the trained apartment price model."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.model import EQUIPMENT, FEATURES, prepare_data, sha256

st.set_page_config(page_title='Morocco apartment prices', page_icon='🏠', layout='wide')
st.title('🏠 Morocco apartment price estimator')
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

predict_tab, data_tab, evaluation_tab = st.tabs(['Estimate', 'Explore data', 'Model evaluation'])
with predict_tab:
    st.write('Estimate an advertised apartment price from its location and characteristics.')
    cities = sorted(df['city_name'].dropna().unique().tolist())
    with st.form('prediction'):
        city = st.selectbox('City', cities)
        area = st.number_input('Surface area (m²)', min_value=1.0, value=100.0, step=1.0)
        rooms = st.number_input('Bedrooms', min_value=0, value=2, step=1)
        baths = st.number_input('Bathrooms', min_value=0, value=1, step=1)
        salons = st.number_input('Living rooms', min_value=0, value=1, step=1)
        values = {}
        if report.get('uses_equipment', False):
            st.write('Equipment (leave unknown if not specified)')
            for label, feature in EQUIPMENT.items():
                values[feature] = st.selectbox(label, ['Unknown', 'Yes', 'No'], key=feature)
        submitted = st.form_submit_button('Estimate price', type='primary')
    if submitted:
        row = {'city_name': city, 'surface_area': area, 'nb_rooms': rooms, 'nb_baths': baths, 'salon': salons}
        row.update({feature: {'Unknown': np.nan, 'Yes': 1.0, 'No': 0.0}[values.get(feature, 'Unknown')]
                    for feature in EQUIPMENT.values()})
        prediction = float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])
        st.metric('Estimated advertised price', f'{prediction:,.0f} MAD')
        city_data = df[df.city_name == city]
        if len(city_data) < 20:
            st.warning(f'Only {len(city_data)} priced records are available for this city; estimates may be less reliable.')
        low, high = df.surface_area.quantile([.01, .99])
        if not low <= area <= high:
            st.warning('This surface area is outside the 1st–99th percentile range of the dataset.')
        st.caption('A statistical estimate, not a valuation. Neighborhood, condition, and other factors are not modeled.')
with data_tab:
    st.subheader('Priced listings in the dataset')
    st.write(f'{len(df):,} rows with a valid positive price. Repeated listings may appear.')
    st.bar_chart(df.city_name.fillna('Unknown').value_counts().head(15))
    st.write('Price distribution (MAD)')
    st.dataframe(df.price.describe(percentiles=[.25, .5, .75, .95]).to_frame('Price').round(0))
    st.caption('Asking prices are historical and may contain errors; this sample is not necessarily representative.')
with evaluation_tab:
    st.subheader('Held-out evaluation')
    st.write(f"Model: {report['selected_model']} · Run: {run_id}")
    a, b, c = st.columns(3)
    a.metric('MAE', f"{report['test_metrics']['mae_dh']:,.0f} MAD")
    b.metric('RMSE', f"{report['test_metrics']['rmse_dh']:,.0f} MAD")
    c.metric('R²', f"{report['test_metrics']['r2']:.3f}")
    st.write(f"Median baseline MAE: {report['baseline_test_metrics']['mae_dh']:,.0f} MAD")
    st.caption('The split groups identical listing links. MAE is an average error, not a prediction interval.')
    if 'error_by_price_band' in report:
        st.write('Error by actual price band')
        st.dataframe(pd.DataFrame(report['error_by_price_band']), hide_index=True)
        st.write('Error by city (small groups are unstable)')
        st.dataframe(pd.DataFrame(report['error_by_city']), hide_index=True)
        st.write('Validation candidates (training folds only)')
        st.dataframe(pd.DataFrame(report['candidates']).T)
