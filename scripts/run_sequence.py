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
#import multiprocessing_on_dil as mp
import multiprocess as mp
from multiprocess import shared_memory
from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures

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

def viz_single_roi(args):
    S = Session(args.dir)
    rois = loader.rois_from_file(S.get_roi_file()) 
    
    window='ROIs_%d'
    rdx = list(map(int,args.roi_index.split(',')))
    windows = [window%r for r in rdx]
    for (idx,w) in enumerate(windows):
        cv2.namedWindow(w)
        cv2.moveWindow(w,300+400*idx,500)
    for s in S.iter_stacks():
        _img = loader.array_from_stack(s)
        for (wdx,r) in enumerate(rdx): 
            roi = rois[r]
            img = _img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
            cv2.imshow(windows[wdx],img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

def viz_extraction(args):
    S = Session(args.dir)
    dx,dy = S.roi_dims() 
    dx = 2*dx
    dy = 2*dy
    nstacks = S.num_stacks() 
    rois = loader.rois_from_file(S.get_roi_file()) 
    
    window='ROIs_%d'
    rdx = list(map(int,args.roi_index.split(',')))
    windows = [window%r for r in rdx]
    Z = [] 
    for (idx,w) in enumerate(windows):
        cv2.namedWindow(w)
        cv2.moveWindow(w,300+400*idx,500)
        fin= S.roi_out.replace('.npy',f'_{idx}.npy')
        Z.append(np.load(fin))

    for i in range(nstacks):
        for (wdx,r) in enumerate(rdx): 
            img = Z[wdx][i,:].reshape(dy,dx)
            img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2BGR)
            cv2.imshow(windows[wdx],img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


def extract_roi(args):
    S = Session(args.dir)
    dx,dy = S.roi_dims() 
    dxy = (2*dx) *(2*dy)
    nstacks = S.num_stacks() 

    rois = loader.rois_from_file(S.get_roi_file()) 

    Z = [np.zeros((nstacks,dxy),dtype=np.uint) for r in rois]
    
    for (idx,s) in tqdm(enumerate(S.iter_stacks()),desc="Stacks extracted:",total=nstacks):
        for wdx,img in iter_extracted_rois(s,rois):
            Z[wdx][idx,:] = img.flatten()  

    for (wdx,roi) in tqdm(enumerate(rois),desc="Saving extractions rois",total=len(rois)):
        fout = S.roi_out.replace('.npy',f'_{wdx}.npy')
        np.save(fout,Z[wdx])

def iter_extracted_rois(stack,rois):
    _img = loader.array_from_stack(stack)
    for (wdx,roi) in enumerate(rois): 
        img = _img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        yield wdx,img

def extract_roi_mp(args):
    S = Session(args.dir)
    dx,dy = S.roi_dims() 
    dxy = (2*dx) *(2*dy)
    nstacks = S.num_stacks() 
    rois = loader.rois_from_file(S.get_roi_file()) 

    nchunks = args.num_jobs
    array_shape = (len(rois),nstacks,dxy) 
    array_name = 'nprois' 
    create_shared_memory_nparray(array_shape,array_name)
   

    futures = []
    with ProcessPoolExecutor(max_workers=nchunks) as executor: 
        for (idx,chunks) in enumerate(S.chunk_stacks(nchunks)):
            futures.append(executor.submit(run_extract_mp,idx,chunks,rois,array_name,array_shape))
    futures, _ = concurrent.futures.wait(futures)
    
    shm = shared_memory.SharedMemory(name=array_name)
    Z = np.ndarray(array_shape, dtype=np.uint, buffer=shm.buf)
    
    for (wdx,roi) in tqdm(enumerate(rois),desc="Saving extractions",total=len(rois)):
        fout = S.roi_out.replace('.npy',f'_{wdx}.npy')
        np.save(fout,Z[wdx,:,:])

    release_shared(array_name)
    

def run_extract_mp(idx,stacks,rois,array_name,array_shape):
    shm = shared_memory.SharedMemory(name=array_name)
    Z = np.ndarray(array_shape, dtype=np.uint, buffer=shm.buf)
    nstacks = len(stacks) 
    for (jdx,s) in tqdm(enumerate(stacks),desc=f"Job {idx}",total=nstacks):
        sdx = nstacks*idx + jdx 
        for wdx,img in iter_extracted_rois(s,rois):
            Z[wdx,sdx,:] = img.flatten()

def create_shared_memory_nparray(array_shape,array_name):
    d_size = np.dtype(np.uint).itemsize * np.prod(array_shape)
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=array_name)
    # numpy array on shared memory buffer
    dst = np.ndarray(shape=array_shape, dtype=np.uint, buffer=shm.buf)
    dst[:] = np.zeros(array_shape,dtype=np.uint)
    print(f'NP SIZE: {(dst.nbytes / 1024) / 1024}')
    return shm


def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()  # Free and release the shared memory block

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

