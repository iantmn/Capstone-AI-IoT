import pickle
import pprint as p
import pandas as pd
import numpy as np
import random as r

from novelty_detection import Novelty_detection
# from Capstone-AI-IoT.Model_Active_Learning.active_learning import Active_learning


def main():
    # model_file = r'../Model_Active_Learning/model_running_50.txt'
    # with open(r'../Model_Active_Learning/model_running_50.txt', 'rb') as f:
    #     model = pickle.load(f)
    # datapd = pd.read_csv(r'../Data Gathering and Preprocessing/features_Walking_scaled.csv')
    # line = np.array(datapd.iloc[1, 3:])
    # lines = np.array(datapd.iloc[2:4, 3:])
    # # print(line)
    # print(model.predict(line.reshape(1, -1)))
    # print(model.predict(lines))
    # p.pprint(datapd)
    nov = Novelty_detection(r'../Model_Active_Learning/model_running_50.txt', r'../Data Gathering and Preprocessing/features_Walking_scaled.csv')
    datapd = pd.read_csv(r'../Data Gathering and Preprocessing/features_Walking_scaled.csv')
    l = len(datapd)
    print(l)
    random_ints = []
    random_ints2 = []
    while len(random_ints) < 50:
        r_int = r.randint(0, l)
        if not r_int in random_ints:
            random_ints.append(r_int)
    while len(random_ints2) < 25:
        r_int = r.randint(0, l)
        if not r_int in random_ints and not r_int in random_ints2:
            random_ints2.append(r_int)    
    nov.detect(random_ints, random_ints2)


if __name__ == "__main__":
    main()