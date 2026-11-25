"""Train a reproducible model and record a local, inspectable experiment."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from src.model import BASE_FEATURES, FEATURES, make_model, prepare_data, sha256


def metrics(y, pred):
    return {'mae_dh': float(mean_absolute_error(y, pred)),
            'rmse_dh': float(np.sqrt(mean_squared_error(y, pred))), 'r2': float(r2_score(y, pred))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='appartements_data.csv')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    data_path = Path(args.data)
    df, groups, dropped = prepare_data(data_path)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, test_idx = next(split.split(df, groups=groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    cv = GroupKFold(n_splits=3)
    experiments = {}
    candidates = [(name, equipped, log) for name in ('random_forest', 'hist_gradient_boosting')
                  for equipped in (False, True) for log in (False, True)]
    for name, equipped, log in candidates:
        key = f"{name}{'_equipment' if equipped else '_base'}{'_log' if log else ''}"
        model = make_model(name, args.seed, equipped, log)
        scores = cross_validate(model, train[FEATURES], train['price'], groups=train_groups, cv=cv,
                                scoring={'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error'})
        experiments[key] = {'cv_mae_dh': float(-scores['test_mae'].mean()),
                            'cv_mae_folds_dh': (-scores['test_mae']).tolist(),
                            'cv_rmse_dh': float(-scores['test_rmse'].mean())}
    selected = min(experiments, key=lambda x: experiments[x]['cv_mae_dh'])
    name, equipped, log = next((n, e, l) for n, e, l in candidates
                               if f"{n}{'_equipment' if e else '_base'}{'_log' if l else ''}" == selected)
    model = make_model(name, args.seed, equipped, log).fit(train[FEATURES], train['price'])
    dummy = DummyRegressor(strategy='median').fit(train[FEATURES], train['price'])
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    run_dir = Path(args.output) / 'runs' / run_id
    run_dir.mkdir(parents=True)
    predictions = model.predict(test[FEATURES])
    errors = pd.DataFrame({'city': test.city_name.fillna('Unknown'), 'actual_mad': test.price,
                           'predicted_mad': predictions, 'link': test.link})
    errors['absolute_error_mad'] = (errors.actual_mad - errors.predicted_mad).abs()
    errors['price_band'] = pd.cut(errors.actual_mad, bins=[0, 500000, 1000000, 2000000, float('inf')],
                                  labels=['0-500k', '500k-1m', '1m-2m', '2m+']).astype(str)
    def grouped_summary(column):
        return errors.groupby(column, dropna=False).agg(count=('actual_mad', 'size'),
                mae_mad=('absolute_error_mad', 'mean')).sort_values('count', ascending=False).round(2).reset_index().to_dict(orient='records')
    errors.sort_values('absolute_error_mad', ascending=False).to_csv(run_dir / 'test_errors.csv', index=False)
    report = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'data_file': str(data_path), 'data_sha256': sha256(data_path), 'seed': args.seed,
              'rows_total': len(df) + dropped, 'rows_missing_or_invalid_price': dropped,
              'rows_train': len(train), 'rows_test': len(test), 'split': 'GroupShuffleSplit by listing link',
              'validation': '3-fold GroupKFold on training listings', 'features': FEATURES if equipped else BASE_FEATURES,
              'uses_equipment': equipped, 'log_target': log,
              'error_by_city': grouped_summary('city'), 'error_by_price_band': grouped_summary('price_band'),
              'candidates': experiments, 'selected_model': selected,
              'test_metrics': metrics(test['price'], predictions),
              'baseline_test_metrics': metrics(test['price'], dummy.predict(test[FEATURES])),
              'versions': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                           'pandas': pd.__version__, 'numpy': np.__version__, 'joblib': joblib.__version__}}
    joblib.dump(model, run_dir / 'model.joblib')
    (run_dir / 'metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    # Update the pointer only after both run artifacts were successfully written.
    (Path(args.output) / 'latest.json').write_text(json.dumps({'run_id': run_id}) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
