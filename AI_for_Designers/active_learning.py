from __future__ import annotations

from AI_for_Designers.Videolabeler import VideoLabeler

import os
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from collections.abc import Sequence
from typing import Any



class ActiveLearning:
    def __init__(self, data_file: str, activity: str, labels: list[str], window_size: float):
        self.preds: np.ndarray | None = None
        self.unpreds: np.ndarray | None = None

        random.seed(42)
        # Get the data and store in datapd
        self.data_file = data_file
        self.datapd = self.get_sensor_data(data_file)
        # Get the amount of features by removing the ID, label and timestamp
        self.number_of_features = self.datapd.shape[1] - 3
        # Determine model in seperate function
        self.model = self.determine_model()
        # PCA
        self.determine_pca()
        # Name of the activity
        self.action_ID = activity
        # Last ID for redo button
        self.lastID = -1
        # Argument is the set of labels that the user predicts
        self.labels = labels
        # A list of the ID's that we have labeled already
        self.labeled_ids = []
        # This is for measuring the functionality/efficiency of the Active learning
        self.gini_margin_acc: list[list[float]] = []

        self.vid = VideoLabeler(labels)
        self.window_size = window_size
        self.html_id = -1

        # X_pool is the dataset that we use for building the model. X_test is to test the model
        self.X_pool, self.X_test, self.y_pool, self.y_test = self.split_pool_test()
        self.X_test = np.array(self.X_test)
        # Remove the labels from the X_pool set
        self.X_pool['label'] = [''] * self.X_pool.shape[0]

        # print(self.testing())

    @property
    def unlabeled_ids(self):
        """We make a property so that when the list of labeled_ids changes we don't have to worry about changing this one."""
        return set(range(self.X_pool.shape[0])) - set(self.labeled_ids)

    @staticmethod
    def determine_model(max_depth: int | None = None) -> RandomForestClassifier:
        """return the selected model for this action classification

        Args:
            max_depth (int, optional): Maximum depth of the chosen model. Defaults to None.

        Returns:
            Object: Returns the model that we want to use for this action classification
        """     
           
        return RandomForestClassifier(max_depth=max_depth, criterion='gini')

    def update_model(self) -> None:
        """This model determines the current average max depth of the trees in the random forest. If the depth has changed drastically since the last check we update the model"""
        # Determine the depth of the current model
        forest = self.determine_model().fit(self.preds[:, 3:], self.preds[:, 1])
        # Multiplication factor of 1.25 so that the tree can grow while actively training but has a limit to prevent overfitting.
        avg_depth = int(sum(estimator.tree_.max_depth for estimator in forest.estimators_)/100*1.25)+1
        self.model = self.determine_model(avg_depth)

    @staticmethod
    def get_sensor_data(data_file: str) -> pd.DataFrame:
        """read and return the datafile from the given path

        Args:
            data_file (str): location of the datafile (csv)

        Returns:
            pd.dataframe: pd dataframe of the datafile
        """        
        return pd.read_csv(data_file)

    def split_pool_test(self) -> list[pd.DataFrame]:
        """splits a dataset into a pool and a test set

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: returns the X_pool, X_test, y_pool, y_test
        """        
        random_state = 42
        test_size = 0.2
        return train_test_split(self.datapd, self.datapd['label'], test_size=test_size, random_state=random_state)

    def training(self, maximum_iterations, random_points: int = 4, cluster_points: int = 1) -> list[str]:
        """the process of training the datapoints, first set starting points, then iterate until you have a certainty

        Args:
            maximum_iterations (_type_): Maximum amount of iterations
            random_points (int, optional): Number of random starting points. Defaults to 3.
            cluster_points (int, optional): Number of clustered starting points. Defaults to 1.
        """

        # Remove prediction point images to prevent flooding when you stop active learning prematurely
        self.remove_pngs()
        # Set randomized starting points       
        self.set_starting_points(random_points)

        # Set the predicted and und predicted sets into new arrays, these will be used further on
        self.preds = np.array(self.X_pool.loc[self.X_pool['label'] != ''])
        self.unpreds = np.array(self.X_pool.loc[self.X_pool['label'] == ''])

        self.clustered_starting_points(cluster_points)

        self.update_model()

        # Set the most ambiguous points iteratively
        self.iterate(maximum_iterations)

        # Save the model as a picle file
        with open(fr'Models/model_{self.action_ID}_{maximum_iterations}.pickle', 'wb') as f:
            pickle.dump(self.model, f)

        # Return the labels, you may find new labels while training
        return self.labels

    def set_starting_points(self, n_samples: int) -> None:
        """Generates training set by selecting random starting points, labeling them, and checking if there's an
        instance of every activity

        Args:
            n_samples (int): _description_
        """             
        # Keep track of what activities we have labeled already
        seen_activities = []  # list of strings
        # Amount of datapoints that we randomly sample
        range_var = n_samples * len(self.labels)
        # self.X_pool.to_csv('test.csv')
        # Generate random points
        for _ in range(range_var):
            # Pick a random point from X_pool
            while True:
                # Set a random id that is in the X_pool and has not yet been labeled
                random_id = random.randint(0, self.datapd.shape[0])
                if random_id not in self.labeled_ids and random_id in self.X_pool['ID']:
                    break
            # Give the timestamp to the identification module but for testing I have automated it
            # got_labeled = self.identify(self.datapd.iloc[random_id]['time'])
            got_labeled = self.identify(random_id)  # for testing
            if got_labeled == 'x':
                # print(np.where(self.datapd.iloc[:, 0] == random_id))
                self.datapd.drop(random_id, 0)
                self.X_pool.drop(random_id, 0)
            # If this label was not accounted for we add it to the set of labels
            else:
                # Redo button:
                if got_labeled == 'r':
                    try:
                        # Remove it from the list that we will identify
                        del self.labeled_ids[-1]
                        del seen_activities[-1]
                        got_labeled = self.identify(random_id)
                    # Catch when you remove the first element
                    except IndexError:
                        raise ValueError("You ain't nah removin' nottin")
                    if got_labeled == 'x':
                        self.datapd.drop(random_id, 0)
                        self.X_pool.drop(random_id, 0)
                        continue
                # Add it to the labels list if it is a new label
                if not (got_labeled in self.labels or got_labeled == 'r'):
                    self.labels.append(got_labeled)
                seen_activities.append(got_labeled)
                self.labeled_ids.append(random_id)
                self.lastID = random_id
        # Fill the X_pool
        for i in range(len(self.labeled_ids)):
            # print(np.where(self.labeled_ids[i]), self.labeled_ids[i])
            self.X_pool.at[self.labeled_ids[i], 'label'] = seen_activities[i]

    def clustered_starting_points(self, n_samples: int = 1) -> None:
        """Find the training point based on a clustering algorithm. You should be able to find at least one sample of each label you predict will be in the dataset.

        Args:
            n_samples (int): Amount of samples that you want from each cluster centre. Keep this value at 1, it is not properly tested. TODO
        """        
        # Amount of clusters that we expect
        # There is a multiplier of 1.5 so that small clusters are not overlooked
        n_clusters = int(len(self.labels) * 1.5) + 1
        # Determine the means
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(self.unpreds[:, 3:])
        cert_indices = []
        # clust_centers = {label:kmeans.cluster_centers_[label] for label in range(len(self.labels))}
        predictions = np.array(kmeans.predict(self.unpreds[:, 3:]))
        X = self.unpreds.copy()
        X[:, 1] = predictions
        for label in range(n_clusters):
            # Select samples with label
            x = X[np.where(X[:, 1] == label)[0], :]
            for _ in range(n_samples):  # For now
                # Transform
                total_dists = np.sum(kmeans.transform(x[:, 3:]) ** 2, axis=1)
                # Add certain samples
                cert_indices.append(x[np.argmin(total_dists), 0])
                # print(cert_indices)
                x = np.delete(x, np.argmin(total_dists), 0)
        for e in cert_indices:
            # got_labeled = self.identify(self.datapd.iloc[min_indices[i]]['time'])
            got_labeled = self.identify(e)  # for testing
            if got_labeled == 'x':
                self.remove_row(e)
            else:
                # Redo button
                if got_labeled == 'r':
                    self.preds = np.delete(self.preds, np.where(self.preds[:, 0] == self.lastID), 0)
                    got_labeled = self.identify(e)
                    if got_labeled == 'x':
                        self.remove_row(e)
                        continue
                    elif got_labeled == 'r':
                        raise ValueError('You sneaky foo. please stop trying to break our code. As punishment you shall be labeling from the start')
                # Add the label to the label list if it is a new label
                if not (got_labeled in self.labels or got_labeled == 'r'):
                    self.labels.append(got_labeled)
                self.labeled_ids.append(e)
                # print(np.where(self.X_pool.iloc[:, 0] == e)[0][0])
                line = self.X_pool.iloc[np.where(self.X_pool.iloc[:, 0] == e)[0][0], :].copy()
                # print(line)
                line.at['label'] = got_labeled
                line = np.array(line).reshape(1, -1)
                # print(line.shape, self.preds.shape)
                self.preds = np.append(self.preds, line, axis=0)
                self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == e), 0)
                self.lastID = e

    def iterate(self, max_iter: int) -> None:
        """This function is the iterative process of active learning. Labeling the most ambiguous points

        Args:
            max_iter (int): maximum number of iterations
        """        
        # This function is the iterative process of active learning. Labeling the most ambiguous points
        iter_num = 0
        while True:
            iter_num += 1
            # find most ambiguous point (find_most_ambiguous_id)
            # label it (set_ambiguous_point)
            # add to training data 
            new_index, margin = self.set_ambiguous_point()
            # Iterate for a decided number of points or until you have a certain margin
            if iter_num >= max_iter or margin > 0.15:
                break

    def set_ambiguous_point(self) -> tuple[int, int]:
        """Lets designer label ambiguous point

        Returns:
            tuple[int, int]: Return the ID that has been labelled and the margin (certainty) of the point that has been labeled.

        """      

        self.html_id = time.time()
        self.remove_pngs()
        # Determine the ID of the most ambiguous datapoint      
        get_id_to_label, margin, les_probs = self.find_most_ambiguous_id()
        # Add it to the IDs that we have labeled
        self.labeled_ids.append(get_id_to_label)
        # Print PCA
        self.print_prediction_point(get_id_to_label)
        # Get what label this ID is supposed to get
        # Just for testing, add les_probs as arg to les_probs if you want these to be printed
        new_label = self.identify(get_id_to_label,
                                  les_probs=les_probs)
        if new_label == 'x':
            self.remove_row(get_id_to_label)
            return get_id_to_label, 0
        else:
            # Redo button
            if new_label == 'r':
                self.preds = np.delete(self.preds, np.where(self.preds[:, 0] == self.lastID), 0)
                new_label = self.identify(get_id_to_label,
                                  les_probs=les_probs)

                if new_label == 'x':
                    self.remove_row(get_id_to_label)
                    return get_id_to_label, 0
                elif new_label == 'r':
                    raise ValueError('You sneaky foo. please stop trying to break our code. As punishment you shall be labeling from the start')
            if not (new_label in self.labels or new_label == 'r'):
                self.labels.append(new_label)
            # Extract the row from the unpredicted array
            t = self.unpreds[self.unpreds[:, 0] == get_id_to_label, :]
            # Label the row
            t[0, 1] = new_label
            # Stack it onto the predicted array
            self.preds = np.vstack((self.preds, t))
            # Delete it from the unpredicted array
            self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == get_id_to_label)[0][0], 0)

            self.lastID = get_id_to_label
            if get_id_to_label in self.unpreds[:, 1]:
                raise ValueError('you did an oopsie')
            
            # Return the label and the margin
            return get_id_to_label, margin

    def identify(self, id, les_probs: dict[str, float] | None = None, process: str = ''):
        """This function will call the identification system from Gijs en Timo, for now it has been automated

        Args:
            id (int): The ID that will be labeled
            les_probs (_type_, optional): When in iterate, this tuple has the probabilities of the current datapoint to each label. When defaulted to None this does not print.

        Returns:
            object: return the return of the videolabeler, so the class that the point is labeled as
        """        
        
        timestamp = self.datapd.iloc[id, 2]
        with open(fr'Preprocessed-data/{self.action_ID}/processed_data_files.txt') as f:
            for line in f:
                split = line.strip().split(',')
                if int(split[1]) <= id <= int(split[2]):
                    video_file = split[3]
                    video_offset = float(split[4])
                    break

        # print(id)
        # print(video_file)
        if les_probs is None:
            return self.vid.labeling(video_file, timestamp, self.window_size, self.html_id, process=process, video_offset=video_offset)
        else:
            return self.vid.labeling(video_file, timestamp, self.window_size, self.html_id, les_probs, process=process, video_offset=video_offset)

        # return input(f'FOR TESTING: enter the selected label, id = {id}\n')

    def find_most_ambiguous_id(self) -> tuple[int, int, list[float]]:
        """Finds the most ambiguous sample. The unlabeled sample with the greatest
            difference between most and second most probably classes is the most ambiguous.
            Returns only the id of this sample

        Raises:
            ValueError: Exception for testing purposes

        Returns:
            tuple[int, int | Any, Any]: returns the id of the most ambiguous sample, the margin and the probabilities
        """        
        try:
            # Fit the model with the datapoints that we have currently labeled.
            self.model.fit(self.preds[:, 3:], self.preds[:, 1])
            # Use this model to get probabilities of datapoints belonging to a certain class.
            sorted_preds = np.sort(self.model.predict_proba(self.unpreds[:, 3:]), axis=1)
            # Basses for the lowest margins
            lowest_margin = 2
            lowest_margin_sample_id: int = 0
            # Append an empty list for the results of this iteration
            self.gini_margin_acc.append([0., 0., 0.])
            # Make a list of the unlabeled ids and sort it
            unlbld = list(self.unlabeled_ids)
            unlbld.sort()
            # Iterate for the length of datapoints that you have not yet labeled
            for i in range(sorted_preds.shape[0]):
                # Subtract from the most certain class the secon to most certain class
                margin = sorted_preds[i, -1] - sorted_preds[i, -2]
                # Is it the lowest?
                if margin < lowest_margin:
                    lowest_margin_sample_id = self.unpreds[i, 0]
                    lowest_margin = margin
                # Add the gini of the datapoint to gini of this iteration
                self.gini_margin_acc[-1][0] += self.gini_impurity_index(list(sorted_preds[i, :]))
            # Make it an average and add the lowest margin
            self.gini_margin_acc[-1][0] /= len(unlbld)
            self.gini_margin_acc[-1][1] = lowest_margin

            les_probs = {}
            for label, prob in zip(self.model.classes_, self.model.predict_proba(
                                   self.unpreds[np.where(self.unpreds[:, 0] == lowest_margin_sample_id)[0], 3:]).tolist()[0]):
                les_probs[label] = prob
            # Oeh fun result get better with more samples Oeh!
            return lowest_margin_sample_id, lowest_margin, les_probs
        # Exception mostly for testing idk if it will every be handydany again
        except ValueError:
            raise ValueError(self.preds)

    @staticmethod
    def gini_impurity_index(list_of_p) -> float:
        """returns the gini: 1 - sum(p^2)

        Args:
            list_of_p (_type_): A list of the probabilities of the to be labeled point belongs to each class

        Returns:
            float: The gini impurity index, to be used for evaluation your model
        """        
        # Return the gini: 1 - sum(p^2)
        return 1 - sum((item * item for item in list_of_p))

    def write_to_file(self) -> str:
        self.unpreds[:, 1] = self.model.predict(self.unpreds[:, 3:])
        nptofile = np.append(self.preds, self.unpreds, axis=0)
        nptofile = nptofile[nptofile[:, 0].argsort()]
        # print(self.preds[:5, :])
        output = fr"{self.data_file.split('.csv')[0]}_AL_predictionss.csv"
        names = np.array([self.datapd.columns])
        np.savetxt(output,
                   np.append(names, nptofile, axis=0), delimiter=",", fmt='%s')

        return output

    def testing(self, n_to_check: int | None = None) -> None:
        """Checks for overwriting based on randomized sampling. To avoid having to make them label the entire test set,
        we ask the designer to confirm n predicted test labels

        Args:
            n_to_check (int | None, optional): Amount of values that you want tested. When None is given, 
            you will iterate through the entire test set (20% of the entire sample size).

        Returns:
            int: error_count and n_to_check
        """       

        self.html_id = -1
        
        # TODO improve:
        # Check for None or numerical size
        if n_to_check is None or n_to_check > len(self.X_test):
            n_to_check = len(self.X_test)
            test_ids = []
            # j is to remember how many samples you deleted
            j = 0
            # Find amount of values that you still need
            while len(test_ids) != n_to_check:
                random_id = random.randint(0, self.datapd.shape[0])
                # Find testing ids
                if random_id in self.X_test[:, 0] and random_id not in test_ids:
                    test_ids.append(random_id)
            predictions = self.model.predict(np.array(self.datapd.iloc[test_ids, 3:]))
            error_count = 0

            # Iterate through the test ids
            for j in range(len(test_ids)):
                result = self.identify(test_ids[j], process='TESTING')
                if result == 'x':
                    j += 1
                elif predictions[j] != result:
                    error_count += 1

            print(f'Error rate: {error_count/(n_to_check-j)} ({n_to_check-j} samples)')
        else:
            # Make sure that 
            i = 0
            # Make sure you always test the amount of values that you gave with n_to_check, even if you delete some sample.
            while i < n_to_check-1:
                test_ids = []
                # Find amount of values that you still need, even if samples have been deleted
                while len(test_ids) != n_to_check-i:
                    random_id = random.randint(0, self.datapd.shape[0])
                    # Find testing ids
                    if random_id in self.X_test[:, 0] and random_id not in test_ids:
                        test_ids.append(random_id)
                predictions = self.model.predict(np.array(self.datapd.iloc[test_ids, 3:]))
                error_count = 0

                # Iterate through the test ids
                for j in range(len(test_ids)):
                    result = self.identify(test_ids[j], process='TESTING')
                    if result == 'x':
                        pass
                    elif predictions[j] != result:
                        error_count += 1
                        i += 1
                    else:
                        i += 1
            print(f'Error rate: {error_count/n_to_check} ({n_to_check} samples)')
        
        # return error_count, n_to_check

    def remove_row(self, id: int) -> None:
        """Removes a row from the data

        Args:
            id (int): id of the row to be removed
        """        
        self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == id)[0][0], 0)
        self.datapd.drop(id, 0)

    def determine_pca(self):
        """Calculates and saves the pca of the data in self.pca
        """        
        pca = PCA(n_components=2, svd_solver='auto')
        self.pca = np.array(pca.fit_transform(self.datapd.iloc[:, 3:]))
        self.pca = np.append(np.array([[i for i in range(len(self.datapd))]]).reshape(-1, 1), self.pca, axis=1)

    def remove_pngs(self):
        directory = r'Plots'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f):
                if 'plot_to_label' in f:
                    os.remove(f)

    def print_prediction_point(self, current_id: int):
        """Creates a file with a plot of the data and the current prediction point

        Args:
            current_id (int): ID of the current prediction point
        """       
        plt.clf()
        plt.scatter(self.pca[:, 1], self.pca[:, 2], c='grey')
        plt.scatter(self.pca[current_id, 1], self.pca[current_id, 2], c='red', marker='x', label='current', s=150)
        for label in self.labels:
            # Pandas made me do it. Fuck pandas
            # lst = list(self.datapd.loc[self.datapd['label'] == label].iloc[:, 0])
            # print(np.where(self.preds[:, 1] == label))
            lst = self.preds[np.where(self.preds[:, 1] == label)[0], :]
            lst = lst[:, 0].tolist()
            # print(lst)
            temp_pca = []
            for e in self.pca:
                # print(int(e[0]), type(e.tolist()), int(e[0]) in lst)
                if int(e[0]) in lst:
                    # print(e)
                    temp_pca.append(e[1:].tolist())
            # print(temp_pca, '\n')

            x = []
            y = []
            for i in range(len(temp_pca)):
                x.append(temp_pca[i][0])
                y.append(temp_pca[i][1])
            plt.scatter(x, y, label=label)
        plt.legend()

        # plt.savefig(f'Plots/plot_to_label.png')
        plt.savefig(f'Plots/plot_to_label_{self.html_id}.png')
        # self.html_id += 1


    def plotting(self) -> None:
        """Plot the gini index, the margin and the test accuracy on every iteration
        """        
        plt.clf()
        plt.plot(np.array(self.gini_margin_acc)[:, :2], label=['gini index', 'margin'])
        plt.xlabel('Iterations [n]')
        plt.ylabel('Uncertainty')
        plt.title('Active learning')
        plt.legend()
        plt.show()
