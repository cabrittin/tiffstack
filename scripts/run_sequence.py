"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import cv2
from tqdm import tqdm
import numpy as np

from pycsvparser import read,write

import tiffstack.loader as loader
from tiffstack.loader import Session
import tiffstack.preprocess as pp

CONFIG = 'config/config.ini'

def set_roi(args):
    S = Session(args.dir) 
    img = S.load_array(0)  
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
    for img in S.iter_stack_array():
        for r in rois:
            cv2.rectangle(img, r[0], r[1], (0, 255, 0), 2)
        
        cv2.imshow(window,img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def viz_single_roi(args):
    S = Session(args.dir)
    rois = loader.rois_from_file(S.get_roi_file()) 
    
    window='ROIs_%d'
    rdx = list(map(int,args.roi_index.split(',')))
    windows = [window%r for r in rdx]
    for (idx,w) in enumerate(windows):
        cv2.namedWindow(w)
        cv2.moveWindow(w,300+400*idx,500)
    for _img in S.iter_stack_array():
        for (wdx,r) in enumerate(rdx): 
            roi = rois[r]
            img = _img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
            cv2.imshow(windows[wdx],img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def extract_roi(args):
    S = Session(args.dir)
    dx,dy = S.roi_dims() 
    dxy = (2*dx) *(2*dy)
    nstacks = S.num_stacks() 

    rois = loader.rois_from_file(S.get_roi_file()) 

    Z = [np.zeros((nstacks,dxy),dtype=np.uint) for r in rois]
    
    for (idx,_img) in tqdm(enumerate(S.iter_stack_array()),desc="Stacks extracted:",total=nstacks):
        for (wdx,roi) in enumerate(rois): 
            img = _img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            Z[wdx][idx,:] = img.flatten()  

    for (wdx,roi) in tqdm(enumerate(rois),desc="Saving extractions rois",total=len(rois)):
        fout = S.roi_out.replace('.npy',f'_{wdx}.npy')
        np.save(fout,Z[wdx])

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
    
    parser.add_argument('-roi_index',
            dest = 'roi_index',
            action = 'store',
            default = "0",
            required = False,
            help = 'ROI index. If multiple rois, should be comma separated; e.g. 1,2,3')


    args = parser.parse_args()
    eval(args.mode + '(args)')

