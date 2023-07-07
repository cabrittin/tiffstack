"""
@name: preprocess.py                       
@description:                  
    functions for preprocessing

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import cv2
from collections import namedtuple


def set_roi(S,idx=0):
    """
    
    Args:
    S: Sequence object
    idx: int, optional (default=0)
    Index of image in sequence to use for ROI setting
    """
    img = S.load(0).pages[idx].asarray()
    window = "ROI"
    dx = 50
    Param = namedtuple("Param", "img window dx rois")
    param = Param(img,window,dx,[])

    cv2.namedWindow(window)
    cv2.moveWindow(window,300,100)
    cv2.setMouseCallback(window,draw_fixed_roi,param)

    while True:
        cv2.imshow(window,img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"): 
            print(param.rois)
            break
    
    cv2.destroyWindow(window)


def draw_fixed_roi(event,x,y,flags,params):
    dx = 50 
    if event == cv2.EVENT_LBUTTONDOWN:
        refpt1 = (x-params.dx,y-params.dx)
        refpt2 = (x+params.dx,y+params.dx)
        cv2.rectangle(params[0], refpt1, refpt2, (0, 255, 0), 2)
        cv2.imshow(params.window,params.img)
        params.rois.append(list(refpt1) + list(refpt2))


     
