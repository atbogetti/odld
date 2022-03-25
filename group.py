import westpa
import logging
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import numpy

log = logging.getLogger(__name__)
log.debug('loading module %r' % __name__)

def kmeans(coords, n_clusters, **kwargs):
    '''Clusters Walkers inside bin according to pcoord value

    Creates a group, which takes the same data format as a bin, and then passes into the
    normal split/merge functions.'''
    # Pass in the bin object instead of the index
    log.debug('using group.kmeans')
    #bin = we_driver.next_iter_binning[ibin]
    groups = dict()
    pcoords = coords
    #for segment in bin:
    #    pcoords.append(segment.pcoord[0])
    X = numpy.array(pcoords)
    X = X.reshape(-1,1)
    km = KMeans(n_clusters=n_clusters[0]).fit(X)   
    cluster_centers_indices = km.cluster_centers_
    labels = km.labels_
    #print('pcoords:', numpy.array(pcoords).flatten())
    print('cluster centers:',cluster_centers_indices.flatten())
    #print('labels:',labels)
    return labels
