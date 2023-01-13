import numpy as np
import pandas as pd
from os import cpu_count

from sklearn.neighbors import LocalOutlierFactor


class NoveltyDetection():
    def __init__(self, data_file):
        self.datapd = pd.read_csv(data_file)

    def detect(self, contamination: int = 0.1) -> list[int]:
        """Function that detects anomalies in the data using LocalOutlierFactor

        Args: 
            contamination (int): The percentage of the dataset that is considered an outlier. Defaults to 0.1.

        Returns:
            list[int]: A list of the ids of the anomalies
        """        
        # Choosing the model LocalOutlierFactor
        clf = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=contamination, n_jobs=int(cpu_count()*3/4))
        # fit and predict the model
        prediction = clf.fit_predict(self.datapd.iloc[:, 3:])
        # Counting the outliers, which are represented with the value -1
        count = 0
        for value in prediction:
            if value == -1:
                count += 1
        # Saving a list of the outliers ids        
        ids = np.where(prediction == -1)[0]
        # Returning the list of the outliers ids
        return ids