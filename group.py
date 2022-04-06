import westpa
import logging
from sklearn import cluster
import numpy

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

def kmeans(coords, n_clusters, splitting):
    X = numpy.array(coords)
    if X.shape[1] == 1:
        X = X.reshape(-1,1)
    km = cluster.KMeans(n_clusters=n_clusters).fit(X)   
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_
    if splitting:
        print("cluster centers:", numpy.sort(cluster_centers_indices))
    return labels

def affinity_propagation(coords, n_clusters, splitting):
    X = numpy.array(coords)
    X = X.reshape(-1,1)
    ap = cluster.AffinityPropagation().fit(X)   
    cluster_centers_indices = ap.cluster_centers_
    labels = ap.labels_
    if splitting:
        print("cluster centers:", numpy.sort(cluster_centers_indices.flatten()))
    return labels

def dbscan(coords, n_clusters, splitting):
    X = numpy.array(coords)
    X = X.reshape(-1,1)
    db = cluster.DBSCAN().fit(X)   
    components = db.components_
    labels = db.labels_
    if splitting:
        print("cluster components:", numpy.sort(components.flatten()))
    return labels
