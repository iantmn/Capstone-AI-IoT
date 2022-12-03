# ---------------------------------------------- #
# Document for testing which models we are going to use. Exploration is being done and this document is not structured in a way that is suitable for production.
# ---------------------------------------------- #

# ------ import ------ #
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV as GSCV
from util import computeFeatureImportance
import csv

np.random.seed(42)

# ------ Settings ------ #

n_clusters = 2

# supervised learning
do_knn = False
do_svc = False
do_rf = False

# clustering
do_kmeans_plot = True

# gridsearch
do_gridsearch_svc = False
do_gridsearch_rf = False

# feature importance
do_feature_importance = True

# ------ Data import ------ #
x = []
with open('Data Gathering and Preprocessing/features_Walking.txt') as csvfile:
    reader = csv.reader(csvfile)
    
    for row in reader:
        x.append(row)
        
# print(x)
x = np.array(x)
# print(x.shape)

# ------ train, test split ------ #

train, test = train_test_split(x, train_size=0.8)

# ------ x, y split ------ #

le = LabelEncoder()
le.fit(train[:, 0:1])
print(le.classes_)

y_train = le.transform(train[:, 0:1])
x_train = train[:, 1:]

y_test = le.transform(test[:, 0:1])
x_test = test[:, 1:]


# ------ PCA ------ #

pca = PCA(2)
print(x_train.shape)
df = pca.fit_transform(x_train)
df_test = pca.fit_transform(x_test)


if do_kmeans_plot:
    # ------ Training KMeans ------ #

    model = KMeans(n_clusters=n_clusters)
    model.fit(df)
    label = model.labels_
    # print(label)

    # ------ controid ------ #

    centroids = model.cluster_centers_
    u_labels = np.unique(label)

    # ------ plot ------ #
    # for i, data in enumerate(x):
    #     print(f"{data[0]=}, {data[1]=}, {label[i]=}")
    #     plt.scatter(data[0], data[1], label=label[i])

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].title.set_text('Model')
    axs[0, 0].scatter(df[:, :1], df[:, 1:], c=label)
    axs[0, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')

    # ------ prediction test data ------ #

    # Make predictions on the test data
    pred = model.predict(df_test)

    axs[0, 1].title.set_text('new data points')
    axs[0, 1].scatter(df_test[:, :1], df_test[:, 1:], c='red')
    axs[0, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')

    # create second plot which show new points whichout prediction
    axs[1, 0].title.set_text('New data on model')
    axs[1, 0].scatter(df[:, :1], df[:, 1:], c=label)
    axs[1, 0].scatter(df_test[:, :1], df_test[:, 1:], c='red')
    axs[1, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')

    # create third plot which show the predictions of the new points
    axs[1, 1].title.set_text('result')
    axs[1, 1].scatter(df[:, :1], df[:, 1:], c=label)
    axs[1, 1].scatter(df_test[:, :1], df_test[:, 1:], c=pred)
    axs[1, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
    plt.legend()
    plt.show()


    fig, axs = plt.subplots(2)
    axs[0].title.set_text('model result')
    axs[0].scatter(df[:, :1], df[:, 1:], c=label)
    axs[0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
    axs[1].title.set_text('Actual result')
    axs[1].scatter(df[:, :1], df[:, 1:], c=y_train, label=('stairs_up', 'stairs_down'))
    axs[1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='x')
    plt.legend()
    plt.show()

# ------ knn ------ #

if do_knn:
    knn = KNN(n_neighbors=3)
    knn.fit(x_train, y_train)

    y_pred_train = knn.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = knn.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"knn: {accuracy_train=}, {accuracy_test=}")

# ------ svc ------ #

if do_svc:
    svc = SVC()
    svc.fit(x_train, y_train)

    y_pred_train = svc.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = svc.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"svc: {accuracy_train=}, {accuracy_test=}")
    
# ------ random forrest ------ #

if do_rf:
    rf = RF()
    rf.fit(x_train, y_train)

    y_pred_train = rf.predict(x_train)
    accuracy_train = accuracy_score(y_train, y_pred_train)

    y_pred_test = rf.predict(x_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    print(f"rf: {accuracy_train=}, {accuracy_test=}")
    
# ------ gridsearch svc ------ #
    
if do_gridsearch_svc:
    parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10, 100]}
    model = SVC()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch rf:")
    print(clf.best_params_)
    print(clf.best_score_)
    
# ------ gridsearch rf ------ #

if do_gridsearch_rf:
    parameters = {'n_estimators':[1, 10, 100, 1000], 'max_depth':[None], 'min_samples_split':[2, 4, 8]}
    model = RF()
    clf = GSCV(model, parameters, verbose=3)
    clf.fit(x_train, y_train)
    print("------------------")
    print("Best parameters for gridsearch rf:")
    print(clf.best_params_)
    print(clf.best_score_)
    
# ------ feature importance ------ #
    
if do_feature_importance:
    x_train_df = pd.DataFrame(x_train, columns=['x_1','x_2','x_3', 'y_1','y_2','y_3', 'z_1','z_2','z_3'])
    imp = computeFeatureImportance(x_train_df, y_train)
    print(imp)