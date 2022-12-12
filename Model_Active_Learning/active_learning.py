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
        # Get the data and store in datapd
        self.data_file = data_file
        self.datapd = self.get_sensor_data(data_file)
        # Get the amount of features by removing the ID, label and timestamp
        self.number_of_features = self.datapd.shape[1]-3
        # Determine model in seperate function
        self.model = self.determine_model()
        # Argument is the set of labels that the user predicts
        self.set_of_labels = set_of_labels
        # A list of the ID's that we have labeled already
        self.labeled_ids = []
        # This is for measuring the functionality/efficiency of the Active learning
        self.gini_margin_acc: list[list[float]] = []

        # X_pool is the dataset that we use for building the model. X_test is to test the model
        self.X_pool, self.X_test, self.y_pool, self.y_test = self.split_pool_test()
        self.X_test = np.array(self.X_test)
        # Remove the labels from the X_pool set
        self.X_pool['label'] = ['']*self.X_pool.shape[0]

        # Train the model with active learning
        self.training()

        # Plot the gini index, the margin and the test accuracy on every iteration
        plt.plot(self.gini_margin_acc, label=['gini index', 'margin', 'test accuracy'])
        plt.xlabel('Iterations [n]')
        plt.ylabel('Magnitude')
        plt.title('Active learning')
        plt.legend()
        plt.show()
        
    @property
    def unlabeled_ids(self):
        """We make a property so that when the list of labeled_ids changes we don't have to worry about changing this one.
        TODO: stash the set"""
        return set(range(self.X_pool.shape[0])).difference(set(self.labeled_ids))
    
    def determine_model(self):
        """return the selected model for this action classification""" 
        return RandomForestClassifier(max_depth=9, criterion='gini')
        
    @staticmethod
    def get_sensor_data(data_file: str):
        """read and return the datafile from the given path"""
        return pd.read_csv(data_file)

    def split_pool_test(self):
        # Parameters for the split
        random_state = 42
        test_size = 0.2
        # return to X_pool, X_test, y_pool, y_test
        return train_test_split(self.datapd, self.datapd['label'], test_size=test_size, random_state=random_state)

    def training(self):
        """the process of training the datapoints, first set starting points, then iterate untill you have a certainty""" 
        # Set randomized starting points       
        self.set_starting_points()
        # Set the predicted and und predicted sets into new arrays, these will be used further on
        self.preds = np.array(self.X_pool.loc[self.X_pool['label'] != ''])
        self.unpreds = np.array(self.X_pool.loc[self.X_pool['label'] == ''])
        # Set the most ambiguous points iteratively
        self.iterate(1900)

    def set_starting_points(self):
        """Generates training set by selecting random starting points, labeling them, and checking if there's an instance of every activity"""
        # Keep track of what activities we have labeled already
        seen_activities = [] # list of strings
        # Amount of datapoints that we randomly sample
        range_var = 4 * len(self.set_of_labels)
        # Generate random points
        for i in range(range_var):
            # Pick a random point from X_pool
            while True:
                # Set a random id that is in the X_pool and has not yet been labeled
                random_id = random.randint(0, self.X_pool.shape[0])
                if random_id not in self.labeled_ids and random_id in self.X_pool['ID']:
                    break
            # Add the ID to the list
            self.labeled_ids.append(random_id)
            # Give the timestamp to the identification module but for testing I have automated it
            # got_labeled = self.identify(self.datapd.iloc[random_id]['time'])
            got_labeled = self.identify(random_id)  # for testing
            # If this label was not accounted for we add it to the set of labels
            if got_labeled not in self.set_of_labels:
                self.set_of_labels.add(got_labeled)
            seen_activities.append(got_labeled)

        # The determined number of random points were executed. We now set random points as long as not all predicted classes were found.
        print('first stage is done!')
        # keep adding points until every activity is in the training set
        while not len(set(seen_activities)) == len(self.set_of_labels):
            while True:
                # We again check if this label is in the pool and if it has not yet been classified.
                random_id = random.randint(0, self.X_pool.shape[0])
                if random_id not in self.labeled_ids and random_id in self.X_pool['ID']:
                    break
            # This all is the same as above
            self.labeled_ids.append(random_id)
            # got_labeled = self.identify(self.datapd.iloc[random_id]['time'])
            got_labeled = self.identify(random_id)
            if got_labeled not in self.set_of_labels:
                self.set_of_labels.add(got_labeled)
            seen_activities.append(got_labeled)

        # We have found a sample of all the labels that we expected
        print('second stage is done!')
        # Randomized phase is done
        # Give labels to the ID's in the pandaset
        for i in range(len(self.labeled_ids)):
            self.X_pool.at[self.labeled_ids[i], 'label'] = seen_activities[i]

    def iterate(self, max_iter):
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
            if iter_num >= max_iter:
                break

            # TODO: Sophie dit is jou idee idk wat je hiermee van plan was
            # show designer plot and performance: ask if they want to stop, continue, or retrain on new samples
            # self.plot_model(f'Iteration {iter_num}', new_index = new_index)
            # question = input("Examine the plot. Enter C if you want to continue, R if your performance is not improving, or S if you are satisfied with this models' performance")

    def set_ambiguous_point(self) -> int:
        """Lets designer label ambiguous point"""          
        # Determine the ID of the most ambiguous datapoint      
        get_id_to_label, margin = self.find_most_ambiguous_id()
        # Add it to the IDs that we have labeled
        self.labeled_ids.append(get_id_to_label)
        # Get what label this ID is supposed to get
        new_label = self.identify(get_id_to_label)  # just for testing

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

    def identify(self, id):
        """This function will call the the identification system from Gijs en Timo, for now it has been automated"""
        # time.sleep(0.2)
        # print(id)
        if 'old' in self.data_file or 'time_features' in self.data_file:
            if id < 91:
                return 'stairs_up'
            elif id < 182:
                return 'stairs_down'
            else:
                return 'walking' 
        else:
            if id < 361:
                return 'stairs_up'
            elif id < 722:
                return 'stairs_down'
            else:
                return 'walking' 
        # return input(f'FOR TESTING: enter the selected label, id = {id}\n')

    def find_most_ambiguous_id(self):
        """Finds the most ambiguous sample. The unlabeled sample with the greatest
            difference between most and second most probably classes is the most ambiguous.
            Returns only the id of this sample"""
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
            # Add the accuracy, this is only for a nice plot and can be deleted afterwards.
            self.gini_margin_acc[-1][2] = accuracy_score(self.model.predict(self.X_test[:, 3:]), self.y_test)
            # Oeh fun result get better with more samples Oeh!
            print(self.gini_margin_acc[-1])
            return lowest_margin_sample_id, lowest_margin
        # Exception mostly for testing idk if it will every be handydany again
        except ValueError:
            # self.X_pool.to_csv('xpool doet raar.csv')
            # print(preds)
            raise ValueError(self.preds)

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

    @staticmethod
    def gini_impurity_index(list_of_p):
        # Return the gini: 1 - sum(p^2)
        return 1-sum((item*item for item in list_of_p))
        
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

    def check_test_set(self):
        """Checks for overfitting based on randomized sampling. To avoid having to make them label the entire test set, we ask 
        the designer to confirm n predicted test labels"""
        test_ids = []
        n_to_check = 30
        while len(test_ids) != n_to_check:
            while True:
                random_id = random.randint(0, self.datapd.shape[0])
                if random_id in self.X_test['ID'] and random_id not in test_ids:
                    test_ids.append(random_id)
                    break
        random_samples = self.X_test['ID'][test_ids]
        predictions = self.model.predict(random_samples)
        error_count = 0
        for i in range(len(test_ids)):
            if predictions[i] != self.identify(self.X_test.at[test_ids[i], 'ID']):
            # if predictions[i] != self.identify(self.datapd.iloc[random_id]['time'])
                error_count += 1
        return error_count

    def add_unseen_activity

