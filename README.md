# Meanshift
Meanshift?

#### Mean shift clustering aims to discover “blobs” in a smooth density of samples. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to eliminate near-duplicates to form the final set of centroids.

##### now we have code :mag:

```python
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
```

you can change the bandwidth, with this work your nodes can see each other from a diffrent distance

I hope this article will be useful to you.
