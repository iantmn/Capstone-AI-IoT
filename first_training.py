# ------ import ------ #

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ------ Settings ------ #

n_clusters = 4

# ------ Data import ------ #

x = 0.3 * np.random.randn(1000, 3)
print(x)

# ------ Data split ------ #

train, test = train_test_split(x, train_size=0.8)

# ------ PCA ------ #

pca = PCA(2)
df = pca.fit_transform(train)
df_test = pca.fit_transform(test)

# ------ Training KMeans ------ #

model = KMeans(n_clusters=n_clusters)
model.fit(df)
label = model.predict(df)
print(label)

# ------ controid ------ #

centroids = model.cluster_centers_
u_labels = np.unique(label)

# ------ plot ------ #
# for i, data in enumerate(x):
#     print(f"{data[0]=}, {data[1]=}, {label[i]=}")
#     plt.scatter(data[0], data[1], label=label[i])

fig, axs = plt.subplots(3)

axs[0].scatter(df[:, :1], df[:, 1:], c=label)
axs[0].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='s')

# ------ prediction test data ------ #

# Make predictions on the test data
pred = model.predict(df_test)

# create second plot which show new points whichout prediction
axs[1].scatter(df[:, :1], df[:, 1:], c=label)
axs[1].scatter(df_test[:, :1], df_test[:, 1:], c='red')
axs[1].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='s')

# create third plot which show the predictions of the new points
axs[2].scatter(df[:, :1], df[:, 1:], c=label)
axs[2].scatter(df_test[:, :1], df_test[:, 1:], c=pred)
axs[2].scatter(centroids[:,0] , centroids[:,1] , s = 80, c="black", marker='s')
plt.show()