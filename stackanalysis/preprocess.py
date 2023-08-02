"""
@name: preprocess.py                         
@description:                  
Functions for preprocessing data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import ndimage

import numpy as np

def pca_local_sequence_diff(Z):
    D = np.diff(Z,axis=0)
    return PCA(n_components=2).fit_transform(D.T)
   

def segment_sequence(Z,mask_switch=0):
    kmeans = KMeans(init="k-means++", n_clusters=2, n_init=4)
    kmeans.fit(Z)
    p = kmeans.predict(Z)
     
    if mask_switch == 1:
        p = 1 - p
    #else:
    #    p1max = Z[p==1].max()
    #    p0max = Z[p==0].max()
    #    if p0max > p1max: p = 1 - p
    
    return p


