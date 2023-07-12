"""
@name: run_analysis.py                         
@description:                  
    Analyze tiffstack extractions

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import os
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import cv2
from tqdm import tqdm
import numpy as np
import multiprocess as mp
from multiprocess import shared_memory
from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import ndimage

from pycsvparser import read,write

import tiffstack.loader as loader
from tiffstack.loader import Session
from stackanalysis import preprocess as pp

def create_mask(args):
    S = Session(args.dir)
    rois = loader.rois_from_file(S.get_roi_file()) 
    for idx in tqdm(range(len(rois)),desc='ROIs processed'):
        fin= S.roi_out.replace('.npy',f'_{idx}.npy')
        Z = np.load(fin)
        reduced_data = pp.pca_local_sequence_diff(Z) 
        p = pp.segment_sequence(reduced_data)
        fout = os.sep.join([S.ext_dir,f'mask_{idx}.npy'])
        np.save(fout,p) 


        fig,ax = plt.subplots(1,1,figsize=(5,5))
        ax.scatter(reduced_data[:,0],reduced_data[:,1],s=2,c=p,cmap='bwr') 
        fout = os.sep.join([S.perf_dir,f'segmentation_{idx}.svg'])
        plt.savefig(fout) 

    """
    dx,dy = S.roi_dims() 
    dx = 2*dx
    dy = 2*dy
    nstacks = S.num_stacks() 
    mask = p.reshape(dy,dx)
    mask = ndimage.binary_dilation(mask,iterations=5).astype(mask.dtype)
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(mask,cmap='gray')
    plt.show()
    window = 'Segmented'
    for i in range(nstacks):
        img = Z[i,:].reshape(dy,dx)
        img = img * mask 
        img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        cv2.imshow(window,img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    """


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('dir',
                        action = 'store',
                        help = 'Directory path')

    parser.add_argument('--roi_index',
            dest = 'roi_index',
            action = 'store',
            default = "0",
            required = False,
            help = 'ROI index. If multiple rois, should be comma separated; e.g. 1,2,3')
    
    parser.add_argument('-n','--num_jobs',
            dest = 'num_jobs',
            action = 'store',
            default = 2,
            type=int,
            required = False,
            help = 'Number of parallel jobs')



    args = parser.parse_args()
    eval(args.mode + '(args)')

