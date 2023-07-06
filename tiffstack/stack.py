"""
@name: stack.py                       
@description:                  
    Class object for using tiff stacks


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2023-07              
"""

from tifffile import TiffFile


class Stack(object):
    """
    Class for storing tiff stack information

    Attributes:
    -----------
    fname :: str
      data file name
    tiff :: TiffFile
      tiff data
    width :: int
      Width of image in pixels
    height :: int
      Height of image in pixels
    length :: int
      Number of pages in tiff
    pxmin :: int
      Miniumum pixel value across all tiff images
    pxmax :: int
      Maximum pixel value across all tiff images
    pxlut :: dict
      Lookup table to convert 16-bit to 8-bit
    scale :: float
      Scaling for visual display
    sample :: list
      List if tiff page indicies. Contains subsample that will be analyzed
    mean :: float
      Mean pixel intensity for images in sample.

    Methods:
    --------
    open():
      Opens tiff file if not already open
    close():
      Closes tiff file if not already closed. Useful for clearing memory
    preprocess():
      Makes lookup table to convert 16-bit image to 8-bit image, by computing
      min and max pixel values in the tiff sequence. 
    mean_pixel_intensity(self): --- REMOVE
      Computes mean pixel intensity for images in self.sample
    randomly_sample(sample):
      Randomly samples images used for analysis
    get_uint8_page(idx):
      Returns numpy array of page idx in unit8 format
    """

    def __init__(self,fname):
        self.fname = fname
        self.tif = None
        self.width = None
        self.height = None
        self.length = None
        self.pxmin = 2**16
        self.pxmax = 0
        self.pxlut = None
        self.sample = None
        self.scale = 0.5
        
    def open(self):
        """
        Opens TiffFile object
        """
        if not self.tif:
            self.tif = TiffFile(self.fname)
            self.width = int(self.tif.pages[0].tags['ImageWidth'].value)
            self.height = int(self.tif.pages[0].tags['ImageLength'].value)
            self.length = len(self.tif.pages)
            self.sample = range(self.length)
        else:
            print('Tiff file already open!')

    def close(self):
        """
        Closes TiffFile object. Useful for releasing memory.
        """
        if self.tif:
            self.tif = None
            self.width = None
            self.height = None
            self.length = None
            self.sample = None
        else:
            print('Tiff file already closed!')
        
    def preprocess(self):
        """
        Makes lookup table to convert 16-bit image to 8-bit image 
        """
        for page in self.tif.pages:
            frame = page.asarray()
            self.pxmin = min(self.pxmin,frame.min())
            self.pxmax = max(self.pxmax,frame.max())
        
        self.pxlut = np.concatenate([
            np.zeros(self.pxmin, dtype=np.uint16),
            np.linspace(0,255,self.pxmax - self.pxmin).astype(np.uint16),
            np.ones(2**16 - self.pxmax, dtype=np.uint16) * 255
            ])

    def map_uint16_to_uint8(self,img):
        """
        Maps image from uint16 to uint8
        """
        return self.pxlut[img].astype(np.uint8)
 
