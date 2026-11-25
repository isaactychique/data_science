"""Produce a non-destructive audit of the source dataset."""
import argparse
import json
from pathlib import Path
import pandas as pd
from src.model import NUMERIC, prepare_data, sha256


def audit(path):
    raw = pd.read_csv(path, na_values=['None'])
    data, groups, dropped = prepare_data(path)
    raw_numeric = raw[NUMERIC].apply(pd.to_numeric, errors='coerce')
    return {
        'data_sha256': sha256(path), 'rows': len(raw), 'valid_positive_prices': len(data),
        'invalid_or_missing_prices': dropped, 'missing_by_column': raw.isna().sum().astype(int).to_dict(),
        'duplicate_link_rows': int(raw['link'].dropna().duplicated().sum()),
        'unique_valid_listing_groups': int(groups.nunique()),
        'potential_anomalies_not_removed': {
            'surface_area_zero_or_negative': int((raw_numeric.surface_area <= 0).sum()),
            'surface_area_above_1000': int((raw_numeric.surface_area > 1000).sum()),
            'bathrooms_above_10': int((raw_numeric.nb_baths > 10).sum()),
            'bedrooms_above_10': int((raw_numeric.nb_rooms > 10).sum()),
            'living_rooms_above_10': int((raw_numeric.salon > 10).sum()),
        },
        'price_quantiles_mad': {str(k): float(v) for k, v in data.price.quantile([0, .25, .5, .75, .95, .99, 1]).items()},
        'top_cities_valid_price': data.city_name.fillna('Unknown').value_counts().head(15).astype(int).to_dict(),
        'note': 'Thresholds flag records for review, not automatic removal or proof of errors.'}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='appartements_data.csv')
    parser.add_argument('--output', default='artifacts/data_audit.json')
    args = parser.parse_args()
    report = audit(args.data)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))
