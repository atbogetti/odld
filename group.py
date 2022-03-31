import westpa
import logging
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import numpy

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

def kmeans(coords, n_clusters, splitting):
    X = numpy.array(coords)
    X = X.reshape(-1,1)
    km = KMeans(n_clusters=n_clusters).fit(X)   
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_
    if splitting:
        print("cluster centers:", numpy.sort(cluster_centers_indices.flatten()))
    return labels
