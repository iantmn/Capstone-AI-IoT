from __future__ import annotations

from AI_for_Designers.Videolabeler import VideoLabeler

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
        self.preds = None
        self.unpreds = None

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
        # Argument is the set of labels that the user predicts
        self.labels = labels
        # A list of the ID's that we have labeled already
        self.labeled_ids = []
        # This is for measuring the functionality/efficiency of the Active learning
        self.gini_margin_acc: list[list[float]] = []

        self.vid = VideoLabeler(labels)
        self.window_size = window_size

        # X_pool is the dataset that we use for building the model. X_test is to test the model
        self.X_pool, self.X_test, self.y_pool, self.y_test = self.split_pool_test()
        self.X_test = np.array(self.X_test)
        # Remove the labels from the X_pool set
        self.X_pool['label'] = [''] * self.X_pool.shape[0]

        # print(self.testing())

    @property
    def unlabeled_ids(self):
        """We make a property so that when the list of labeled_ids changes we don't have to worry about changing this one.
        TODO: stash the set"""
        return set(range(self.X_pool.shape[0])).difference(set(self.labeled_ids))

    @staticmethod
    def determine_model(max_depth: int = None):
        """return the selected model for this action classification

        Args:
            max_depth (int, optional): Maximum depth of the chosen model. Defaults to None.

        Returns:
            Object: Returns the model that we want to use for this action classification
        """     
           
        return RandomForestClassifier(max_depth=max_depth, criterion='gini')

    def update_model(self):
        """This model determines the current average max depth of the trees in the random forest. If the depth has changed drastically since the last check we update the model"""
        # Determine the depth of the current model
        self.model.estimator_.max_depth
        forest = self.determine_model().fit(self.preds)
        avg_depth = sum(estimator.tree_.max_depth for estimator in forest.estimators_)/100
        print(avg_depth)
        self.model = self.determine_model(avg_depth)

    @staticmethod
    def get_sensor_data(data_file: str):
        """read and return the datafile from the given path

        Args:
            data_file (str): location of the datafile (csv)

        Returns:
            pd.dataframe: pd dataframe of the datafile
        """        
        return pd.read_csv(data_file)

    def split_pool_test(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """splits a dataset into a pool and a test set

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: returns the X_pool, X_test, y_pool, y_test
        """        
        # Parameters for the split
        random_state = 42
        test_size = 0.2
        # return to X_pool, X_test, y_pool, y_test
        return train_test_split(self.datapd, self.datapd['label'], test_size=test_size, random_state=random_state)

    def training(self, maximum_iterations, random_points: int = 3, cluster_points: int = 1) -> None:
        """the process of training the datapoints, first set starting points, then iterate until you have a certainty

        Args:
            maximum_iterations (_type_): Maximum amount of iterations
            random_points (int, optional): Number of random starting points. Defaults to 3.
            cluster_points (int, optional): Number of clustered starting points. Defaults to 1.
        """     

        # Set randomized starting points       
        self.set_starting_points(random_points)

        # Set the predicted and und predicted sets into new arrays, these will be used further on
        # print(self.X_pool.loc[self.X_pool['label'] != ''])
        self.preds = np.array(self.X_pool.loc[self.X_pool['label'] != ''])
        self.unpreds = np.array(self.X_pool.loc[self.X_pool['label'] == ''])

        self.clustered_starting_points(cluster_points)
        # pd.to_csv('hello?', self.datapd)

        # Set the most ambiguous points iteratively
        self.iterate(maximum_iterations)
        # Write to csv file
        # self.write_to_file()

        with open(fr'Models/model_{self.action_ID}_{maximum_iterations}.txt', 'wb') as f:
            pickle.dump(self.model, f)

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
        for i in range(range_var):
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
                print(np.where(self.datapd.iloc[:, 0] == random_id))
                self.datapd.drop(random_id, 0)
                self.X_pool.drop(random_id, 0)
            # If this label was not accounted for we add it to the set of labels
            else:
                if got_labeled not in self.labels:
                    self.labels.append(got_labeled)
                seen_activities.append(got_labeled)
                # Add the ID to the list
                self.labeled_ids.append(random_id)

        # The determined number of random points were executed. We now set random points as long as not all predicted
        # classes were found.
        print('first stage is done!')

        # keep adding points until every activity is in the training set
        # We added samples as the centre of each cluster so we don't really need this bit anymore
        # while not len(set(seen_activities)) == len(self.labels):
        #     # print(len(set(seen_activities)), len(self.labels))
        #     while True:
        #         # We again check if this label is in the pool and if it has not yet been classified.
        #         random_id = random.randint(0, self.X_pool.shape[0])
        #         if random_id not in self.labeled_ids and random_id in self.X_pool['ID']:
        #             break
        #     # This all is the same as above
        #     self.labeled_ids.append(random_id)
        #     # got_labeled = self.identify(self.datapd.iloc[random_id]['time'])
        #     got_labeled = self.identify(random_id)
        #     if got_labeled not in self.labels:
        #         self.labels.append(got_labeled)
        #     seen_activities.append(got_labeled)

        # We have found a sample of all the labels that we expected
        print('second stage is done!')
        # Randomized phase is done
        # Give labels to the ID's in the pandaset
        # print(self.X_pool.iloc[0, :])
        # print(self.X_pool)
        # print(self.X_pool.iloc[self.labeled_ids[0], :])
        for i in range(len(self.labeled_ids)):
            print(np.where(self.labeled_ids[i]), self.labeled_ids[i])
            self.X_pool.at[self.labeled_ids[i], 'label'] = seen_activities[i]
        # self.X_pool.to_csv('test2.csv')
        # self.datapd.to_csv('test3.csv')

    def clustered_starting_points(self, n_samples: int) -> None:
        """_summary_

        Args:
            n_samples (int): _description_
        """        
        # Amount of clusters that we expect
        n_clusters = len(self.labels)
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
            self.labeled_ids.append(e)
            if got_labeled == 'x':
                self.remove_row(e)
            else:
                # print(np.where(self.X_pool.iloc[:, 0] == e)[0][0])
                line = self.X_pool.iloc[np.where(self.X_pool.iloc[:, 0] == e)[0][0], :].copy()
                # print(line)
                line.at['label'] = got_labeled
                line = np.array(line).reshape(1, -1)
                # print(line.shape, self.preds.shape)
                self.preds = np.append(self.preds, line, axis=0)
                self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == e), 0)

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
            # Iterate for a decided number of points or until you have a certain margin or gini
            # TODO: Investigate margin or gini at which you have a good accuracy
            if iter_num >= max_iter or margin > 0.15:
                break

            # TODO: Sophie dit is jou idee idk wat je hiermee van plan was
            # show designer plot and performance: ask if they want to stop, continue, or retrain on new samples
            # self.plot_model(f'Iteration {iter_num}', new_index = new_index)
            # question = input("Examine the plot. Enter C if you want to continue, R if your performance is not "
            #                  "improving, or S if you are satisfied with this models' performance")

    def set_ambiguous_point(self) -> tuple[int, int]:
        """Lets designer label ambiguous point

        Raises:
            ValueError: _description_

        Returns:
            tuple[int, int]: _description_
        """        
        # Determine the ID of the most ambiguous datapoint      
        get_id_to_label, margin, les_probs = self.find_most_ambiguous_id()
        # Add it to the IDs that we have labeled
        self.labeled_ids.append(get_id_to_label)
        # Print PCA
        self.print_predition_point(get_id_to_label)
        # Get what label this ID is supposed to get
        # Just for testing, add les_probs as arg to les_probs if you want these to be printed
        new_label = self.identify(get_id_to_label,
                                  les_probs=les_probs)
        if new_label == 'x':
            self.remove_row(get_id_to_label)
            return get_id_to_label, 0
        else:
            # Extract the row from the unpredicted array
            t = self.unpreds[self.unpreds[:, 0] == get_id_to_label, :]
            # Label the row
            t[0, 1] = new_label
            # Stack it onto the predicted array
            self.preds = np.vstack((self.preds, t))
            # Delete it from the unpredicted array
            self.unpreds = np.delete(self.unpreds, np.where(self.unpreds[:, 0] == get_id_to_label)[0][0], 0)
            if get_id_to_label in self.unpreds[:, 1]:
                raise ValueError('you did an oopsie')
            # Return the label and the margin
            return get_id_to_label, margin

    def identify(self, id, les_probs=None):
        """This function will call the identification system from Gijs en Timo, for now it has been automated

        Args:
            id (int): _description_
            les_probs (_type_, optional): _description_. Defaults to None.

        Returns:
            object: _description_
        """        
        # time.sleep(0.2)
        # print(id)
        # if les_probs is not None:
        #     print(les_probs, les_probs.index(max(les_probs)))
        # if 'old' in self.data_file or 'time_features' in self.data_file:
        #     if id < 91:
        #         return 'stairs_up'
        #     elif id < 182:
        #         return 'stairs_down'
        #     else:
        #         return 'walking' 
        # else:
        #     if id < 543:
        #         return 'stairs_up'
        #     elif id < 1177:
        #         return 'stairs_down'
        #     elif id < 1632:
        #         return 'running'
        #     else:
        #         return 'walking'
        timestamp = self.datapd.iloc[id, 2]
        with open(fr'Preprocessed-data/{self.action_ID}/processed_data_files.txt') as f:
            for line in f:
                split = line.strip().split(',')
                if int(split[1]) <= id <= int(split[2]):
                    video_file = split[3]
                    break

        # print(id)
        # print(video_file)
        if les_probs is None:
            return self.vid.labeling(video_file, timestamp, self.window_size)
        else:
            return self.vid.labeling(video_file, timestamp, self.window_size, les_probs)

        # return input(f'FOR TESTING: enter the selected label, id = {id}\n')

    def find_most_ambiguous_id(self) -> tuple[int, int | Any, Any]:
        """Finds the most ambiguous sample. The unlabeled sample with the greatest
            difference between most and second most probably classes is the most ambiguous.
            Returns only the id of this sample

        Raises:
            ValueError: _description_

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

            les_probs = self.model.predict_proba(
                self.unpreds[np.where(self.unpreds[:, 0] == lowest_margin_sample_id)[0], 3:]).tolist()[0]

            # Add the accuracy, this is only for a nice plot and can be deleted afterwards.
            # self.gini_margin_acc[-1][2] = accuracy_score(self.model.predict(self.X_test[:, 3:]), self.y_test)
            # Oeh fun result get better with more samples Oeh!
            # print(self.gini_margin_acc[-1])
            return lowest_margin_sample_id, lowest_margin, les_probs
        # Exception mostly for testing idk if it will every be handydany again
        except ValueError:
            # self.X_pool.to_csv('xpool doet raar.csv')
            # print(preds)
            raise ValueError(self.preds)

    @staticmethod
    def gini_impurity_index(list_of_p) -> float:
        """returns the gini: 1 - sum(p^2)

        Args:
            list_of_p (_type_): _description_

        Returns:
            float: The gini impurity index
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

    def testing(self, n_to_check: int | None = None) -> int:
        """Checks for overwriting based on randomized sampling. To avoid having to make them label the entire test set,
        we ask the designer to confirm n predicted test labels

        Args:
            n_to_check (int | None, optional): _description_. Defaults to None.

        Returns:
            int: error_count and n_to_check
        """       
        
        # TODO improve:
        if n_to_check is None or n_to_check > len(self.X_test):
            n_to_check = len(self.X_test)
        i = 0
        while i < n_to_check-1:
            test_ids = []
            while len(test_ids) != n_to_check-i:
                random_id = random.randint(0, self.datapd.shape[0])
                if random_id in self.X_test[:, 0] and random_id not in test_ids:
                    test_ids.append(random_id)
            predictions = self.model.predict(np.array(self.datapd.iloc[test_ids, 3:]))
            error_count = 0

            for j in range(len(test_ids)):
                result = self.identify(test_ids[j])
                if result == 'x':
                    pass
                elif predictions[j] != result:
                    error_count += 1
                    i += 1
                else:
                    i += 1
        print(f'Error rate: {error_count/n_to_check} ({n_to_check} samples)')
        
        return error_count, n_to_check

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

    def print_predition_point(self, current_id: int):
        """Creates a file with a plot of the data and the current prediction point

        Args:
            current_id (int): ID of the current prediction point
        """        
        ids = self.labeled_ids
        ids.sort()
        plt.scatter(self.pca[:, 1], self.pca[:, 2], c='grey')
        for e in ids:
            plt.scatter(self.pca[e, 1], self.pca[e, 2], c='blue')
        plt.scatter(self.pca[current_id, 1], self.pca[current_id, 2], c='red', marker='x')

        plt.savefig(f'Plots/plot_to_label_{loremipsum}.png')


    def plotting(self) -> None:
        """Plot the gini index, the margin and the test accuracy on every iteration
        """        
        plt.plot(self.gini_margin_acc, label=['gini index', 'margin', 'test accuracy'])
        plt.xlabel('Iterations [n]')
        plt.ylabel('Magnitude')
        plt.title('Active learning')
        plt.legend()
        plt.show()

    # def iteration_0(self):
    #     X_train = X_pool.iloc[self.labeled_ids]
    #     y_train = y_pool.iloc[self.labeled_ids]
    #     self.model.fit(X_train, y_train)
    #     self.plot_model('Iteration 0')

    # def evaluate_model(self):
    #     """_summary_
    #     """        
    #     '''This function gives possibly relevant evaluation metrics like accuracy, precision, recall and F1 score.'''
    #     y_pred = self.model.predict(self.X_test)
    #     y_true = self.y_test
    #     test_acc = accuracy_score(y_test, y_pred)
    #     print("Test Accuracy : ", test_acc)
    #     print("MCC Score : ", matthews_corrcoef(y_true, y_pred))
    #     print("Classification Report : ")
    #     print(classification_report(self.y_test, y_pred))

    # def plot_model(self, title: str, new_index: bool = False):
    #     """_summary_

    #     Args:
    #         title (str): _description_
    #         new_index (bool, optional): _description_. Defaults to False.
    #     """        
    #     '''Makes a plot. Black points are unlabeled, red points are labeled, star is the most ambiguous point in that iteration.'''
    #     xlabel = 'Dimension 1'
    #     ylabel = 'Dimension 2'
    #     # define data variables
    #     self.X_train = self.X_pool.iloc[self.labeled_ids]
    #     self.y_train = self.y_pool.iloc[self.labeled_ids]
    #     self.X_unk = self.X_pool.iloc[self.unlabeled_ids]
    #     if new_index:
    #         X_new = self.X_pool.iloc[new_index]
    #     # plot points
    #     plt.scatter(X_unk, c='k', marker = '.')
    #     plt.scatter(X_train, y_train, c='r', marker = 'o')
    #     if new_index:
    #         plt.scatter(X_new, c='y', marker="*", s=125)
    #     # axis and title name
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.title(title)
    #     plt.show()

    # def label_test_set(self):
    #     pass
    # predict y_test
    # let user confirm or correct

    # def define_activities(self):
    #     activity_list = []
    #     ask_activity = input("Please input expected activities. Type X when you're done. ")
    #     while ask_activity != "X":
    #         activity_list.append(ask_activity)
    #         ask_activity = input()
    #     for i in range(len(activity_list)):
    #         self.activities[activity_list[i]] = i + 1
    #     print(f"These are your activities: {activity_list}. If you want to make any changes, run this cell again. ")
    # done = input("Type Y if you're done, or N if you want to add more activities. ")
    # if done == "N":
    # ask_activity = input()
    # return self.activities
