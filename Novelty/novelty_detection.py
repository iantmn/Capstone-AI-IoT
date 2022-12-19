import pickle
import numpy as np
import pandas as pd
from os import cpu_count

from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans

from collections.abc import Collection


class Anomaly_detection():
    def __init__(self, model_file, data_file):
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        self.datapd = pd.read_csv(data_file)
        self.labels = self.datapd
        self.closest_to_centres, self.possible_novelties = self.find_closest_to_centres(0.5)

    @property
    def labels(self) -> list[str]:
        return self._labels

    @labels.setter
    def labels(self, pd):
        labels: set[str] = set()
        for i in range(len(pd)):
            labels |= {pd.iloc[i, 1]}
        self._labels = labels

    def detect(self, fit_ids: Collection | None = None, predict_ids: Collection | None = None):
        clf = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=0.1, n_jobs=int(cpu_count()*3/4))
        # if fit_ids is None:
        #     clf.fit(self.datapd.iloc[:, 3:])
        # else:
        #     clf.fit(self.datapd.iloc[fit_ids, 3:])
        # ps = self.datapd.iloc[predict_ids, 1]
        # print(ps)
        # prediction = clf.predict(self.datapd.iloc[predict_ids, 3:])
        # print(prediction)
        # clf.fit(self.closest_to_centres.iloc[:, 3:], self.closest_to_centres.iloc[:, 1])
        # clf.fit(self.datapd.iloc[:, 3:])
        # prediction = clf.predict(self.possible_novelties.iloc[:, 3:])
        prediction = clf.fit_predict(self.datapd.iloc[:, 3:])
        count = 0
        for value in prediction:
            if value == -1:
                count += 1
                
        ids = np.where(prediction == -1)[0]
        # ids = self.datapd.iloc[prediction, :]
        print(ids)

        return ids

    def find_closest_to_centres(self, part):
        n_clusters = len(self.labels)

        data = np.array(self.datapd.iloc[:, 3:])

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans = kmeans.fit(data)

        predictions = np.array(kmeans.predict(data))
        X = np.array(self.datapd).copy()
        X[:, 1] = predictions

        closest_to_centers: list[list[int]] = []
        possible_novelties: list[list[int]] = []
        for label in range(n_clusters):
            x = X[np.where(X[:, 1] == label)[0], :]
            print(x.shape, len(x), int(x.shape[0]*part))
            for _ in range(int(len(x)*part)):
                # print(i)
                # print(x.shape)
                transformed = kmeans.transform(x[:, 3:])**2

                total_dists = np.sum(transformed, axis = 1)
                closest_to_centers.append(x[np.argmin(total_dists), 0])
                # print(x.shape)
                x = np.delete(x, np.argmin(total_dists), 0)
                # print(x.shape)
            # print(x.shape)
            for i in x:
                possible_novelties.append(i[0])
            # print(len(possible_novelties), len(possible_novelties[0]))

        print(len(closest_to_centers), len(possible_novelties))
            
        return self.datapd.iloc[closest_to_centers, :], self.datapd.iloc[possible_novelties, :]

        


    # def clustered_starting_points(self, n_samples):
    #     # Amount of clusters that we expect
    #     n_clusters = len(self.set_of_labels)
    #     # Determine the meanse
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.unpreds[:,3:])
    #     cert_indices = []
    #     # clust_centers = {label:kmeans.clustercenters[label] for label in range(len(self.set_of_labels))}
    #     predictions = np.array(kmeans.predict(self.unpreds[:,3:]))
    #     X = self.unpreds.copy()
    #     X[:, 1] = predictions
    #     for label in range(nclusters):
    #         # Select samples with label
    #         x = X[np.where(X[:,1]==label)[0], :]
    #         for  in range(n_samples): # For now
    #             # Transform
    #             total_dists = np.sum(kmeans.transform(x[:, 3:])**2, axis=1)
    #             # Add certain samples
    #             cert_indices.append(x[np.argmin(total_dists), 0])
    #             # print(cert_indices)
    #             x = np.delete(x, x[np.argmin(total_dists), 0])
    #     for e in cert_indices:
    #         # got_labeled = self.identify(self.datapd.iloc[min_indices[i]]['time'])
    #         got_labeled = self.identify(e)  # for testing
    #         self.labeled_ids.append(e)
    #         # print(np.where(self.X_pool.iloc[:, 0] == e)[0][0])
    #         line = self.X_pool.iloc[np.where(self.X_pool.iloc[:, 0] == e)[0][0], :].copy()
    #         # print(line)
    #         line.at['label'] = got_labeled
    #         line = np.array(line).reshape(1, -1)
    #         # print(line.shape, self.preds.shape)
    #         self.preds = np.append(self.preds, line, axis=0)
    #         self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == e), 0)