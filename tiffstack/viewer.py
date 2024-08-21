"""
@name: tiffstack.viewer                        
@description:                  
Module for viewing tiff stacks in a VIM like editor

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import cv2

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
                return 0
            
            elif key == ord('k'): 
                idx = min(max_i,idx+1)
            
            elif key == ord('j'): 
                idx = max(0,idx-1)
           
            elif key == ord('u'):
                idx = max(0,idx-large_iter)
            
            elif key == ord('f'):
                idx = min(max_i,idx+large_iter)

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
 
            
            S.user_update(key,jdx,idx)

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



