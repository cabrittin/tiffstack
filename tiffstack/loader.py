"""
@name: loader.py                         
@description:                  

Class object for loading tiff filess

Maintains a buffer cache

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2023-07         
"""

from configparser import ConfigParser,ExtendedInterpolation
import os
import glob
import re
import numpy as np

from tifffile import TiffFile

from pycsvparser.read import parse_file

class Session(object):
    def __init__(self,paths):
        if len(paths) == 1: 
            self.stacks = stacks_from_path(paths)
        else:
            paths.sort(key=natural_keys)
            self.stacks = paths
        tif = tif_from_stack(self.stacks[0])
        self.height = tif.pages[0].shape[0] 
        self.width = tif.pages[0].shape[1] 
        self.depth = len(tif.pages)

    def __len__(self):
        return len(self.stacks)

    def get_stack(self,idx):
        return self.stacks[idx]
    
    def iter_stacks(self):
        for s in self.stacks:
            yield s

    def chunk_stacks(self,nchunks):
        return split_n(self.stacks,nchunks)
    

def split_n(sequence, num_chunks):
    chunk_size, remaining = divmod(len(sequence), num_chunks)
    for i in range(num_chunks):
        begin = i * chunk_size + min(i, remaining)
        end = (i + 1) * chunk_size + min(i + 1, remaining)
        yield sequence[begin:end]

def tif_from_stack(fname):
    return TiffFile(fname)

def array_from_stack(fname,zdx=0):
    return tif_from_stack(fname).pages[zdx].asarray()

def image_to_rgb(image):
    ndims = len(image.shape)
    if ndims < 3:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    return image

def image_to_gray(image):
    ndims = len(image.shape)
    if ndims > 2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return image

def array_16bit_to_8bit(a):
    # If already 8bit, return array
    if a.dtype == 'uint8': return a
    a = np.array(a,copy=True)
    display_min = np.amin(a)
    display_max = np.amax(a)
    a.clip(display_min, display_max, out=a)
    a -= display_min
    np.floor_divide(a, (display_max - display_min + 1) / 256,
                    out=a, casting='unsafe')
    return a.astype('uint8') 


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def format_path(L):
    ext = L.cfg['extension']
    path = os.sep.join([L.dname,'*'+ext])
    return path

def format_glob_path(directory,extension):
    return os.sep.join([directory,'*'+extension])
    
def stacks_from_path(path,order_stacks=True):
    stacks = glob.glob(path)
    if order_stacks: stacks.sort(key=natural_keys)
    return stacks
