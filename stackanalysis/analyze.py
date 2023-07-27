"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np
from sklearn.decomposition import PCA

def proportion_pixel_change(Z,mask,zthresh=2):
    Z = Z.astype(np.int32)
    Z0 = Z[:,mask == 0]
    Z1 = Z[:,mask == 1]
     
    D0 = abs(np.diff(Z0,axis=0))
    mu = D0.mean(axis=1)
    sig = D0.std(axis=1)

    mu = np.tile(mu,(Z1.shape[1],1)).T
    sig = np.tile(sig,(Z1.shape[1],1)).T
    
    D1 = np.diff(Z1,axis=0)
    D1 = np.divide(np.abs(D1) - mu,sig)
    
    D1[D1<zthresh] = 0
    D1[D1>0] = 1
    p = D1.mean(axis=1)
    
    return p

def pixel_above_background(Z,mask):
    Z = Z.astype(np.int32)
    Z0 = Z[:,mask == 0]
    Z1 = Z[:,mask == 1]
    
    mu = Z0.mean(axis=1)
    sig = Z0.std(axis=1)
    
    mu = np.tile(mu,(Z1.shape[1],1)).T
    sig = np.tile(sig,(Z1.shape[1],1)).T

    X = np.divide(Z1 - mu,sig)

    return X

def frame_pixel_histogram(Z,mask):
    Z = Z.astype(np.int32)
    Z1 = Z[:,mask == 1]
    
    H = np.zeros((Z1.shape[0],255))
    for i in range(Z1.shape[0]):
        H[i,:] = np.histogram(Z1[i,:],bins=255,range=(0,255),density=True)[0]
    
    return H

def pixel_pca(Z,mask,n_components=50,trange=None):
    Z = Z.astype(np.int32)
    Z1 = Z[:,mask == 1]
    pca = PCA(n_components=n_components)
    if trange is not None: Z1 = Z1[trange[0]:trange[1],:]
    pca.fit(Z1) 
    return pca
