"""
@name: tiffstack.viewer                        
@description:                  
Module for viewing tiff stacks in a VIM like editor

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import numpy as np
import cv2

MAX_PIXEL = 2**16 - 1

class Viewer:
    def __init__(self,D):
        """
        Parameters:
        ----------
        D: Generic dataset class
        """
        
        self.D = D
        self.jdx = 0
        self.idx = 0
    
    def load_dims(self):
        self.sequence_size = self.D.get_sequence_size()
        self.stack_size = self.D.get_stack_size()
        self.shape = self.D.get_shape()
 
    def preprocess(self):
        self.load_dims()
        self.pxmin = self.D.get_pxmin()
        self.pxmax = self.D.get_pxmax()
        self.pxlut = compute_lut(self.pxmin,self.pxmax)
    
    def on_close(self):
        pass

    def init_window(self):
        self.win ='Stack'
        cv2.namedWindow(self.win)
        cv2.moveWindow(self.win,800,500)
        self.update_title()

    def update_title(self):
        #wtitle = self.D.get_window_title(self.jdx,self.idx)
        wtitle = self.get_current_title() 
        cv2.setWindowTitle(self.win,wtitle)
    
    def get_current_title(self):
        return 'Window'

    def get_start_indicies(self):
        return self.jdx,self.idx

    def load_stack(self,jdx):
        self.jdx = jdx
    
    def load_slice(self,idx):
        self.idx = idx

    def get_sequence_size(self):
        return self.sequence_size

    def get_stack_size(self):
        return self.stack_size
    
    def map_uint16_to_uint8(self,img):
        """
        Maps image from uint16 to uint8
        """
        return self.pxlut[img].astype(np.uint8)
    
    def display(self,idx):
        self.idx = idx 
        self.update_title()
        img  = self.get_current_image()
        return self.map_uint16_to_uint8(img)
    
    def get_current_image(self):
        return self.D.get_image(self.idx,self.jdx)

    def user_update(self,key,sequence_jdx,stack_idx):
        jdx = sequence_jdx
        idx = stack_idx
        
        if key == ord('b'):
            self.pxmax = max(self.pxmin,self.pxmax-100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('t'):
            self.pxmax = min(MAX_PIXEL,self.pxmax+100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('v'):
            self.pxmax = max(self.pxmin,self.pxmax-1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('r'):
            self.pxmax = min(MAX_PIXEL,self.pxmax+1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
 
        elif key == ord('w'):
            self.pxmin = min(self.pxmax,self.pxmin+100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('x'):
            self.pxmin = max(0,self.pxmin-100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('e'):
            self.pxmin = min(self.pxmax,self.pxmin+1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('c'):
            self.pxmin = max(0,self.pxmin-1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
       
        jdx,idx = self._user_update(key)
        
        return jdx,idx
    
    def _user_update(self,key):
        return 

def compute_lut(pxmin,pxmax):
    pxlut = np.concatenate([
        np.zeros(pxmin, dtype=np.uint16),
        np.linspace(0,255,pxmax - pxmin).astype(np.uint16),
        np.ones(MAX_PIXEL - pxmax, dtype=np.uint16) * 255
        ])
    
    return pxlut

def update_display(T):
    cout = f'''
        Sequence length: {T.sequence_size}

        Keys:
        -----
        Quit: q 

        Next (prev) timepoint: l(h)
        Next (prev) z-slice: k(j)
        
        Raise max pixel +100 (+1): t(r)
        Lower max pixel -100 (-1): b(v)
        Raise min pixel +100 (+1): w(e)
        Lower min pixel -100 (-1): x(c)

        Timelapse:
        ----------
        Max pixel = {T.pxmax} 
        Min pixel = {T.pxmin} 
        
        '''
 
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    
    loop = range(1, len(cout) + 1)
    for idx in reversed(loop): print(LINE_UP, end=LINE_CLEAR)

    print(cout,end='\r')


def image_looper(S,large_iter=100,reset_idx=False):
    jdx,idx = S.get_start_indicies()  
    max_j = S.get_sequence_size() - 1
    while True:
        S.load_stack(jdx)
        max_i = S.get_stack_size() - 1
        if reset_idx: idx = 0
        while True:
            img = S.display(idx)
            cv2.imshow(S.win,img)
            key = cv2.waitKey(0) & 0xFF 

            if key == ord('q'): 
                S.on_close() 
                return 0
            
            elif key == ord('k'): 
                if idx == max_i: 
                    idx = 0
                else:
                    idx += 1
                S.load_slice(idx)
            
            elif key == ord('j'): 
                S.load_slice(idx)
                if idx == 0: 
                    idx = max_i
                else:
                    idx -= 1
                S.load_slice(idx)
            
            elif key == ord('h'):
                if jdx == 0: 
                    jdx = max_j
                else:
                    jdx -= 1
                break
            
            elif key == ord('l'):
                if jdx == max_j: 
                    jdx = 0
                else:
                    jdx += 1
                break
            
            elif key == ord('z'):
                jdx = 0
                break

            elif key == ord('u'):
                jdx = (jdx + large_iter) % max_j
                break 

            elif key == ord('f'):
                jdx = (jdx-large_iter) % max_j
                break
            
            jdx,idx = S.user_update(key,jdx,idx)

def stack_saver(S,fout):
    size = S.display_size
    #fourcc = cv2.VideoWriter_fourcc(*'XVID') 
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    result = cv2.VideoWriter(fout,fourcc,10, size)
    
    for idx in range(S.get_stack_size()):
        img = S.display(idx)
        result.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    result.release()
    cv2.destroyAllWindows()
    print(f'Video saved to {fout}')


class Stack(object):
    def __init__(self,*args,**kwargs):
        self.sequence_size = 0
        self.stack_size = 0

    def get_sequence_size(self):
        return self.sequence_size
    
    def get_stack_size(self):
        return self.stack_size
    
    def load_stack(self,jdx):
        pass

    def display(self,idx):
        pass

    def user_update(self,key,sequence_jdx,stack_idx):
        pass

    def destructor(self):
        pass



