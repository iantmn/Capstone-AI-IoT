# ------ import ------ #
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ------ Settings ------ #

n_clusters = 4

# ------ Data import ------ #

x = 0.3 * np.random.randn(1000, 20)
print(x.shape)

# ------ train, test split ------ #

train, test = train_test_split(x, train_size=0.8)

# ------ x, y split ------ #

le = LabelEncoder()
le.fit(train[:, 0])
print(le.classes_)

y_train = le.transform(train[:, 0])
x_train = train[:, 1:]

y_test = le.transform(test[:, 0])
x_test = test[:, 1:]

# ------ PCA ------ #

pca = PCA(2)
df = pca.fit_transform(train)
df_test = pca.fit_transform(test)

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

axs[0, 0].scatter(df[:, :1], df[:, 1:], c=label)
axs[0, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='s')

# ------ prediction test data ------ #

# Make predictions on the test data
pred = model.predict(df_test)

axs[0, 1].scatter(df_test[:, :1], df_test[:, 1:], c='red')

# create second plot which show new points whichout prediction
axs[1, 0].scatter(df[:, :1], df[:, 1:], c=label)
axs[1, 0].scatter(df_test[:, :1], df_test[:, 1:], c='red')
axs[1, 0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='s')

# create third plot which show the predictions of the new points
axs[1, 1].scatter(df[:, :1], df[:, 1:], c=label)
axs[1, 1].scatter(df_test[:, :1], df_test[:, 1:], c=pred)
axs[1, 1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='s')
plt.show()


# ------ KNN ------ #

knn = KNN(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred_train = knn.predict(x_train)
accuracy_train = accuracy_score(y_train, y_pred_train)

y_pred_test = knn.predict(x_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"{accuracy_train=}, {accuracy_test=}")