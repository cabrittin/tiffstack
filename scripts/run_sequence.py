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

from pycsvparser import read,write

import tiffstack.loader as loader
from tiffstack.loader import Session
import tiffstack.preprocess as pp

CONFIG = 'config/config.ini'

def set_roi(args):
    S = Session(args.dir) 
    img = S.load_array(0)  
    window = 'ROI'
    dx = S.cfg.getint('roi','dx')
    rois = []
    pp.set_roi(img,window,dx,rois)
    fout = S.get_roi_file()
    write.from_list(fout,rois)


def viz_roi(args):
    S = Session(args.dir)
    _rois = read.into_list(S.get_roi_file(),multi_dim=True) 
    rois = [[(int(r[0]),int(r[1])),(int(r[2]),int(r[3]))] for r in _rois]

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
    _rois = read.into_list(S.get_roi_file(),multi_dim=True) 
    rois = [[(int(r[0]),int(r[1])),(int(r[2]),int(r[3]))] for r in _rois]
    
    roi = rois[0]
    
    window='ROIs'
    cv2.namedWindow(window)
    cv2.moveWindow(window,300,100)
    for _img in S.iter_stack_array():
        
        img = _img[roi[0][1]:roi[1][1],roi[0][0]:roi[1][0]]
        
        cv2.imshow(window,img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break



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

