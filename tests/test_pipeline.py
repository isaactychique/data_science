import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.model import FEATURES, make_model, prepare_data


class PipelineTest(unittest.TestCase):
    def test_missing_target_group_split_and_unknown_city(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.csv'
            rows = []
            for i in range(20):
                rows.append({'city_name': 'Rabat', 'surface_area': 60 + i, 'nb_rooms': 2,
                             'nb_baths': 1, 'salon': 1, 'price': f'{500000+i*1000} DH',
                             'link': f'listing-{i // 2}'})
            rows.append({**rows[0], 'price': None})
            pd.DataFrame(rows).to_csv(path, index=False)
            data, groups, dropped = prepare_data(path)
            self.assertEqual(dropped, 1)
            train, test = next(GroupShuffleSplit(test_size=.2, random_state=42).split(data, groups=groups))
            self.assertTrue(set(groups.iloc[train]).isdisjoint(groups.iloc[test]))
            model = make_model('random_forest').fit(data.iloc[train][FEATURES], data.iloc[train]['price'])
            example = data.iloc[test[:1]][FEATURES].copy()
            example.loc[:, 'city_name'] = 'Unseen city'
            example.loc[:, 'salon'] = np.nan
            self.assertTrue(np.isfinite(model.predict(example)).all())

if __name__ == '__main__':
    unittest.main()
