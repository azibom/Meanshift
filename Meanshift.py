# import requirements
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
import numpy as np
# load data
iris = load_iris()
# set model
x = iris.data
MeanShift = MeanShift(bandwidth=2)
MeanShift.fit(x)
# give the label and scatter
labels = MeanShift.labels_
cluster_center = MeanShift.cluster_centers_

n_cluster = len(np.unique(labels))

plt.scatter(x[:,0], x[:,2], c=labels)
plt.scatter(cluster_center[:,0], cluster_center[:,2], marker='x', s=150, linewidth=5, zorder=10)
plt.show()

