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
from scipy import ndimage
import seaborn as sns

from pycsvparser import read,write

import tiffstack.loader as loader
from tiffstack.loader import Session
from stackanalysis import preprocess as pp
from stackanalysis import analyze as az

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


def viz_mask(args):
    S = Session(args.dir)
    dx,dy = S.roi_dims() 
    dx = 2*dx
    dy = 2*dy
    nstacks = S.num_stacks() 
    
    window='Mask_%d'
    rdx = list(map(int,args.roi_index.split(',')))
    windows = [window%r for r in rdx]
    for (idx,w) in enumerate(windows):
        cv2.namedWindow(w)
        cv2.moveWindow(w,300+400*idx,500)
    
    mask = []
    Z = [] 
    for (idx,r) in enumerate(rdx):
        fin= S.roi_out.replace('.npy',f'_{idx}.npy')
        Z.append(np.load(fin)) 

        fin = os.sep.join([S.ext_dir,f'mask_{idx}.npy']) 
        mask.append(np.load(fin)) 
    
    for i in range(nstacks):
        for (wdx,r) in enumerate(rdx): 
            img = Z[wdx][i,:]
            img = img * mask[wdx] 
            img = cv2.cvtColor(img.reshape(dy,dx).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            cv2.imshow(windows[wdx],img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def compute_pixel_change(args):
    S = Session(args.dir)
    nstacks = S.num_stacks()
    pixel_change_thresh = S.cfg.getint('analysis','pixel_change_thresh')
    wsize = S.cfg.getint('analysis','smoothing_kernel_size')  
    rois = loader.rois_from_file(S.get_roi_file()) 
    P = np.zeros((len(rois),nstacks-1))
    Pm = np.zeros((len(rois),P.shape[1] - (wsize - 1))) 
    for idx in tqdm(range(len(rois)),desc='ROIs processed'):
        fin= S.roi_out.replace('.npy',f'_{idx}.npy')
        Z = np.load(fin) 
        fin = os.sep.join([S.ext_dir,f'mask_{idx}.npy']) 
        mask = np.load(fin)
       
        p = az.proportion_pixel_change(Z,mask,zthresh=pixel_change_thresh)
        P[idx,:] = p

        pm = np.convolve(p,np.ones(wsize),"valid") / float(wsize)
        Pm[idx,:] = pm

        t = np.arange(len(p))
        tm = np.arange(len(pm)) + 0.5*wsize

        fig,ax = plt.subplots(1,2,figsize=(30,5))
        ax[0].hist(p,bins=50)
        ax[0].set_ylabel('counts per bin') 
        ax[0].set_xlabel('proportion pixel change')
        ax[1].plot(t,p,'-k')
        ax[1].plot(tm,pm,'-r')
        ax[1].set_xlabel('time index')
        ax[1].set_ylabel('proportion pixel change')

        fout = os.sep.join([S.perf_dir,f'prop_pixel_change_{idx}.svg'])
        plt.savefig(fout) 
    
    fout = os.sep.join([S.ext_dir,f'prop_pixel_change.npy'])
    np.save(fout,P)
    

    fig,ax = plt.subplots(1,1,figsize=(15,5))
    sns.heatmap(ax=ax,data=Pm,cmap='viridis') 
    ax.set_xticklabels([])
    ax.set_xlabel('time index')
    ax.set_ylabel('ROIs')
    fout = os.sep.join([S.perf_dir,f'prop_pixel_change_summary.png'])
    plt.savefig(fout,dpi=300)

def compute_pixel_distribution(args):
    S = Session(args.dir)
    nstacks = S.num_stacks()
    rois = loader.rois_from_file(S.get_roi_file()) 
    idx = 3 
    for idx in tqdm(range(len(rois)),desc='ROIs processed'):
        fin= S.roi_out.replace('.npy',f'_{idx}.npy')
        Z = np.load(fin) 
        fin = os.sep.join([S.ext_dir,f'mask_{idx}.npy']) 
        mask = np.load(fin)

        P = az.pixel_above_background(Z,mask)
        mu = P.mean(axis=1)
        t = np.arange(P.shape[0])
        fig,ax = plt.subplots(1,2,figsize=(30,5))
        ax[0].hist(P.flatten(),bins=50)
        ax[0].set_ylabel('counts per bin') 
        ax[0].set_xlabel('pixels relative to background')
        ax[1].plot(t,mu,'-k')
        ax[1].set_xlabel('time index')
        ax[1].set_ylabel('average pixels relative to background')

        fout = os.sep.join([S.perf_dir,f'pixel_relative_to_background_{idx}.svg'])
        plt.savefig(fout) 
 

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

