"""
@name: extract_training.py                       
@description:                  
    Extract and format training data for pyobjectdetect package

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import os
import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import cv2
from tqdm import tqdm
import numpy as np
#import multiprocessing_on_dil as mp
import multiprocess as mp
from multiprocess import shared_memory
from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures
import matplotlib.pyplot as plt
import random

from pycsvparser import read,write

import tiffstack.loader as loader
from tiffstack.loader import Session
import tiffstack.preprocess as pp

CONFIG = 'config/config.ini'

def set_roi(args):
    S = Session(args.dir) 
    fin = S.stacks[0] 
    img = loader.array_from_stack(fin) 
    
    window = 'ROI'
    dx,dy = S.roi_dims() 
    rois = []
    pp.set_roi(img,window,dx,dy,rois)
    fout = S.get_roi_file()
    write.from_list(fout,rois)

def viz_roi(args):
    S = Session(args.dir)
    rois = loader.rois_from_file(S.get_roi_file()) 

    window='ROIs'
    cv2.namedWindow(window)
    cv2.moveWindow(window,300,100)
    for s in S.iter_stacks():
        img = loader.array_from_stack(s)
        for r in rois:
            cv2.rectangle(img, r[0], r[1], (0, 255, 0), 2)
        
        cv2.imshow(window,img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


def build_masks(args):
    S = Session(args.dir)
    dx,dy = S.roi_dims() 
    dx = 2*dx
    dy = 2*dy

    rois = loader.rois_from_file(S.get_roi_file()) 
    
    Z = np.zeros((S.height,S.width),dtype=np.uint8)
    
    for (idx,[(x0,y0),(x1,y1)]) in enumerate(rois):
        fin = os.sep.join([S.ext_dir,f'mask_{idx}.npy']) 
        mask = np.load(fin).reshape(dy,dx).astype(np.uint8)
        mask = mask * idx
        Z[y0:y1,x0:x1] = mask
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8)) 
    ax.set_axis_off()
    ax.imshow(Z)

    plt.show()

def build_masks(S):
    dx,dy = S.roi_dims() 
    dx = 2*dx
    dy = 2*dy

    rois = loader.rois_from_file(S.get_roi_file()) 
    
    Z = np.zeros((S.height,S.width),dtype=np.uint8)
    
    for (idx,[(x0,y0),(x1,y1)]) in enumerate(rois):
        fin = os.sep.join([S.ext_dir,f'mask_{idx}.npy']) 
        mask = np.load(fin).reshape(dy,dx).astype(np.uint8)
        mask = mask * idx
        Z[y0:y1,x0:x1] = mask
    
    return Z 

def run_stack(func):
    def inner(args,**kwargs):
        S = Session(args.dir)
        dx,dy = S.roi_dims() 
        dx = 2*dx
        dy = 2*dy    
        
        mask = build_masks(S)
        rois = loader.rois_from_file(S.get_roi_file()) 
        
        bdx = [0,2,5] 

        window='Blanks'
        cv2.namedWindow(window)
        cv2.moveWindow(window,300,100)
        for s in S.iter_stacks():
            img = loader.array_from_stack(s)
            M = mask.copy()
            func(img,blanks=bdx,rois=rois,roi_size=(dy,dx),mask=M)
            
            cv2.imshow(window,img)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
             
            fig, ax = plt.subplots(1, 1, figsize=(16, 8)) 
            ax.set_axis_off()
            ax.imshow(M)

            plt.show()
   
    return inner

@ run_stack
def viz_blanks(img,blanks=None,rois=None,roi_size=None,mask=None,**kwargs):
    blank_image(img,blanks,rois,roi_size,mask=mask)

@ run_stack
def viz_rotate(img,blanks=None,rois=None,roi_size=None,mask=None,**kwargs):
    blank_image(img,blanks,rois,roi_size,mask=mask)
    rotate_rois(img,rois,blanks=blanks,mask=mask)

@ run_stack
def viz_flip(img,blanks=None,rois=None,roi_size=None,mask=None,**kwargs):
    blank_image(img,blanks,rois,roi_size,mask=mask)
    flip_rois(img,rois,blanks=blanks,mask=mask)

@ run_stack
def viz_swap(img,blanks=None,rois=None,roi_size=None,n_swaps=10,mask=None,**kwargs):
    blank_image(img,blanks,rois,roi_size,mask=mask)
    for _ in range(n_swaps): swap_rois(img,rois,blanks=blanks,mask=mask)

@ run_stack
def viz_translate(img,blanks=None,rois=None,roi_size=None,n_trans=10,mask=None,**kwargs):
    blank_image(img,blanks,rois,roi_size,mask=mask)
    rois = rois[:] 
    for _ in range(n_trans):
        img,rois = translate_rois(img,rois,blanks=blanks,mask=mask)

@ run_stack
def viz_all_translations(img,blanks=None,rois=None,roi_size=None,n_swaps=10,n_trans=10,mask=None,**kwargs):
    blank_image(img,blanks,rois,roi_size,mask=mask)
    rois = rois[:] 
    for _ in range(n_trans):
        img,rois = translate_rois(img,rois,blanks=blanks,mask=mask)
    
    for _ in range(n_swaps): swap_rois(img,rois,blanks=blanks,mask=mask)
    flip_rois(img,rois,blanks=blanks,mask=mask)
    rotate_rois(img,rois,blanks=blanks,mask=mask)

def blank_image(img,blanks,rois,roi_size,mask=None):
    dy,dx = roi_size 
    blank = img[:dy,:dx]
    for bdx in blanks:  
        (x0,y0),(x1,y1) = rois[bdx]
        img[y0:y1,x0:x1] = blank
        if mask is not None: 
            mask[y0:y1,x0:x1] = 0

def inplace_roi_mod(func):
    def inner(img,rois,blanks=[],mask=None):
        for idx,((x0,y0),(x1,y1)) in enumerate(rois):
            if idx in blanks: continue
            _img = img[y0:y1,x0:x1]
            img[y0:y1,x0:x1],mod_val = func(img[y0:y1,x0:x1])
            if mask is not None:
                mask[y0:y1,x0:x1],mod_val = func(mask[y0:y1,x0:x1],mod_val=mod_val)
                
    return inner

@inplace_roi_mod
def rotate_rois(*args,mod_val=-1,**kwargs):
    if mod_val < 0: 
        mod_val = np.random.randint(low=1,high=4)
    return np.rot90(args[0],mod_val),mod_val

@inplace_roi_mod
def flip_rois(*args,mod_val=-1,**kwargs):
    if mod_val < 0: 
        mod_val = np.random.randint(low=0,high=2)
    return np.flip(args[0],mod_val),mod_val

def swap_rois(img,rois,blanks=[],mask=None):
    rdx = [i for i in range(len(rois)) if i not in blanks]
    (idx,jdx) = random.sample(rdx,k=2)
    swap_regions(img,rois[idx],rois[jdx]) 
    if mask is not None: swap_regions(mask,rois[idx],rois[jdx]) 

def swap_regions(img,p0,p1):
    _img = img.copy()
    (x0_0,y0_0),(x0_1,y0_1) = p0
    (x1_0,y1_0),(x1_1,y1_1) = p1
    img0 = _img[y0_0:y0_1,x0_0:x0_1]
    img1 = _img[y1_0:y1_1,x1_0:x1_1]
    img[y0_0:y0_1,x0_0:x0_1] = img1
    img[y1_0:y1_1,x1_0:x1_1] = img0

def translate_rois(img,rois,blanks=[],mask=None):
    rdx = [i for i in range(len(rois)) if i not in blanks]
    idx = random.choice(rdx)
    (x0,y0),(x1,y1) = rois[idx]
    dx = x1 - x0
    dy = y1 - y0
    
    xpos = np.random.randint(low=0,high=img.shape[1]-dx)
    ypos = np.random.randint(low=0,high=img.shape[0]-dy)
    p0 = (xpos,ypos)
    p1 = (xpos+dx,ypos+dy)
    _rois = [rois[i] for i in rdx]
    if no_roi_conflict(img,p0,p1,rois): 
        swap_regions(img,rois[idx],(p0,p1))
        if mask is not None: swap_regions(mask,rois[idx],(p0,p1)) 
        rois[idx] = [p0,p1]
    
    return img,rois

def no_roi_conflict(img,p0,p1,rois):
    Z = np.zeros(img.shape)
    for (x0,y0),(x1,y1) in rois:
        Z[y0:y1,x0:x1] = 1

    T = np.zeros(img.shape)
    T[p0[1]:p1[1],p0[0]:p1[0]] = 1
    
    P = np.multiply(Z,T)
    return P.max() == 0


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

    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')
    

    args = parser.parse_args()
    eval(args.mode + '(args)')

