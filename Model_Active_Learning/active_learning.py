from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections.abc import Sequence
import random
import time


class Active_learning():
    def __init__(self, data_file: str, set_of_labels: Sequence):
        self.data_file = data_file
        self.datapd = self.get_sensor_data(data_file)
        self.number_of_features = self.datapd.shape[1]-3
        self.model = self.determine_model()
        self.set_of_labels = set_of_labels
        self.labeled_ids = []
        self.gini_indices: list[float] = []
        # self.unlabeled_ids = set(e for e in range(0, self.datapd.shape[0]))
        # self.train_indexes = list(range(10))
        # self.unknown_indexes = list(range(10, len_pool))
        self.X_pool, self.X_test, self.y_pool, self.y_test = self.split_pool_test()
        # self.activities = {}
        self.training()
        plt.plot(self.gini_indices)
        plt.xlabel('Iterations [n]')
        plt.ylabel('Gini impurity')
        plt.title('Active learning gini degradation.')
        plt.show()
        
    @property
    def unlabeled_ids(self):
        """TODO: stash the set"""
        return set(range(self.datapd.shape[0])).difference(set(self.labeled_ids))
    
    def determine_model(self):
        """return the selected model for this action classification""" 
        return RandomForestClassifier(max_depth=9, criterion='gini')
        
    @staticmethod
    def get_sensor_data(data_file: str):
        """read and return the datafile from the given path"""
        return pd.read_csv(data_file)

    def split_pool_test(self):
        random_state = 42
        test_size = 0.2
        # print(list(self.datapd.columns[3:]))
        self.datapd['label'] = ['']*self.datapd.shape[0]
        return train_test_split(self.datapd[['ID', 'label'] + list(self.datapd.columns[3:])], 
                                self.datapd['label'], test_size=test_size, random_state=random_state)

    def training(self):
        """the process of training the datapoints, first set starting points, then iterate untill you have a certainty"""        
        self.set_starting_points()
        self.iterate(260)

    def set_starting_points(self):
        """Generates training set by selecting random starting points, labeling them, and checking if there's an instance of every activity"""
        seen_activities = [] # list of strings
        range_var = 4 * len(self.set_of_labels)
        # generate random points
        for i in range(range_var):
            # pick a random point from X_pool
            while True:
                random_id = random.randint(0, self.datapd.shape[0])
                if random_id not in self.labeled_ids:
                    break
            self.labeled_ids.append(random_id)
            # got_labeled = self.identify(self.datapd.iloc[random_id]['time'])
            got_labeled = self.identify(random_id)  # for testing
            if got_labeled not in self.set_of_labels:
                self.set_of_labels.add(got_labeled)
            seen_activities.append(got_labeled)

        print('first stage is done!')
        # keep adding points until every activity is in the training set
        while not len(set(seen_activities)) == len(self.set_of_labels):
            while True:
                random_id = random.randint(0, self.datapd.shape[0])
                if random_id not in self.labeled_ids:
                    break
            self.labeled_ids.append(random_id)
            # got_labeled = self.identify(self.datapd.iloc[random_id]['time'])
            got_labeled = self.identify(random_id)
            if got_labeled not in self.set_of_labels:
                self.set_of_labels.add(got_labeled)
            seen_activities.append(got_labeled)

        print('second stage is done!')
        # Randomized phase is done
        # Give labels to the ID's in the pandaset
        print(self.labeled_ids, seen_activities)
        for i in range(len(self.labeled_ids)):
            # print(self.labeled_ids[i], self.datapd['ID', str(self.labeled_ids[i])])
            self.datapd.at[self.labeled_ids[i], 'label'] = seen_activities[i]
        # self.datapd.to_csv('for testing.csv', index=False)

    def iterate(self, max_iter):
        iter_num = 0
        while True:
            iter_num += 1
            # find most ambiguous point (find_most_ambiguous_id)
            # label it (set_ambiguous_point)
            # add to training data 
            new_index, margin = self.set_ambiguous_point()
            print(new_index)
            if iter_num >= max_iter:
                break

            # show designer plot and performance: ask if they want to stop, continue, or retrain on new samples
            # self.plot_model(f'Iteration {iter_num}', new_index = new_index)
            # question = input("Examine the plot. Enter C if you want to continue, R if your performance is not improving, or S if you are satisfied with this models' performance")

    def set_ambiguous_point(self) -> int:
        """Lets designer label ambiguous point

        Returns:
            int: ID that has been labeled
        """                
        get_id_to_label, margin = self.find_most_ambiguous_id()
        # print(f'id_to_label: {get_id_to_label}, margin: {margin}, gini_index: {self.gini_indices[-1]}')
        self.labeled_ids.append(get_id_to_label)
        # self.datapd.at[get_id_to_label, 'label'] = self.identify(self.datapd.at[get_id_to_label, 'time'])
        self.datapd.at[get_id_to_label, 'label'] = self.identify(get_id_to_label)  # just for testing
        return get_id_to_label, margin

    def identify(self, id):
        # time.sleep(1)
        if id < 91:
            return 'a'
        elif id < 182:
            return 'b'
        else:
            return 'c' 
        # return input(f'FOR TESTING: enter the selected label, id = {id}\n')

    def find_most_ambiguous_id(self):
        '''Finds the most ambiguous sample. The unlabeled sample with the greatest
            difference between most and second most probably classes is the most ambiguous.
            Returns only the id of this sample'''
        preds = np.array(self.datapd.loc[self.datapd['label'] != ''])
        unpreds = np.array(self.datapd.loc[self.datapd['label'] == ''])
        # print(self.datapd.shape, preds.shape)
        self.model.fit(preds[:, 3:], preds[:, 1])
        unpreds = self.model.predict_proba(unpreds[:, 3:])
        # print(unpreds.shape)
        sorted_preds = np.sort(unpreds, axis=1)
        # print(sorted_preds[:5, :])
        lowest_margin = 1
        lowest_margin_sample_id: int = 0
        self.gini_indices.append(0.)
        unlbld = list(self.unlabeled_ids)
        # print(f'lengths: unlbld: {len(unlbld)}, sorted_preds: {sorted_preds.shape}')
        unlbld.sort()
        for i in range(len(unlbld)):
            # idk of unknown_sample['ID'] werkt:
            margin = sorted_preds[i, -1] - sorted_preds[i, -2]
            if margin < lowest_margin:
                # print(i, sorted_preds[i, :], list(unlbld)[i])
                lowest_margin_sample_id = list(unlbld)[i]
                lowest_margin = margin
            self.gini_indices[-1] += self.gini_impurity_index(list(sorted_preds[i, :]))
        self.gini_indices[-1] /= len(unlbld)
        # most_ambiguous = X_pool.iloc[lowest_margin_sample_id]
        return lowest_margin_sample_id, lowest_margin

    def iteration_0(self):
        X_train = X_pool.iloc[self.labeled_ids]
        y_train = y_pool.iloc[self.labeled_ids]
        self.model.fit(X_train, y_train)
        self.plot_model('Iteration 0')

    def evaluate_model(self):
        """_summary_
        """        
        '''This function gives possibly relevant evaluation metrics like accuracy, precision, recall and F1 score.'''
        y_pred = self.model.predict(self.X_test)
        y_true = self.y_test
        test_acc = accuracy_score(y_test, y_pred)
        print("Test Accuracy : ", test_acc)
        print("MCC Score : ", matthews_corrcoef(y_true, y_pred))
        print("Classification Report : ")
        print(classification_report(self.y_test, y_pred))

    def plot_model(self, title: str, new_index: bool = False):
        """_summary_

        Args:
            title (str): _description_
            new_index (bool, optional): _description_. Defaults to False.
        """        
        '''Makes a plot. Black points are unlabeled, red points are labeled, star is the most ambiguous point in that iteration.'''
        xlabel = 'Dimension 1'
        ylabel = 'Dimension 2'
        # define data variables
        self.X_train = self.X_pool.iloc[self.labeled_ids]
        self.y_train = self.y_pool.iloc[self.labeled_ids]
        self.X_unk = self.X_pool.iloc[self.unlabeled_ids]
        if new_index:
            X_new = self.X_pool.iloc[new_index]
        # plot points
        plt.scatter(X_unk, c='k', marker = '.')
        plt.scatter(X_train, y_train, c='r', marker = 'o')
        if new_index:
            plt.scatter(X_new, c='y', marker="*", s=125)
        # axis and title name
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    @staticmethod
    def gini_impurity_index(list_of_p):
        return 1-sum((item*item for item in list_of_p))

        
    def label_test_set(self):
        pass
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

