import pickle
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor

from collections.abc import Collection

class Novelty_detection():
    def __init__(self, model_file, data_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.datapd = pd.read_csv(data_file)

    def detect(self, fit_ids: Collection | None = None, predict_ids: Collection | None = None):
        clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
        if fit_ids is None:
            clf.fit(self.datapd.iloc[:, 3:])
        else:
            clf.fit(self.datapd.iloc[fit_ids, 3:])
        print('fitted')
        ps = self.datapd.iloc[predict_ids, 1]
        print(ps)
        prediction = clf.predict(self.datapd.iloc[predict_ids, 3:])
        print(prediction)

        
        