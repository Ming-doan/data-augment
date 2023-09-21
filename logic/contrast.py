"""
Writer: Võ Phi Trường
"""

from ._utils import *
import cv2

class SigmoidContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False
        self.randgain = DistRand(range=(5, 20, 1), mode='norm')
        self.randcutoff = DistRand(range=(0.25, 0.75, 0.05), mode='norm')
        
    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        
        # gain=(5, 20) cutoff=(0.25, 0.75)
        gain = self.randgain.rand()
        cutoff = self.randcutoff.rand()
        # Convert the Image to numpy array
        images = images.astype(np.float32)
        
        # 
        images = images / 255
        
        # Sigmoid function
        images = 1.0 / (1.0 + np.exp(-gain * (images - cutoff)))
        
        # Convert numpy array to the PIL image
        images = (images * 255).astype(np.uint8)
        # Define your algorithm here 👆
        return images, bboxs


class GammarContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False
        self.randgamma = DistRand(range=(0.5,2.0,0.1),mode='norm')

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        
        # Generate a gamma value
        gamma = self.randgamma.rand()
        # Convert the image to numpy array
        images = images.astype(np.float32)
        #
        images = images/255
        # Gamma function
        images  = images ** gamma
        # Conver numpy array to the PIL Images
        images = (images*255).astype(np.uint8)
        # gamma = (0.5, 2.0)
        # Define your algorithm here 👆
        return images, bboxs


class HistogramContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        # Convert the image to gray
        images = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
        h, s, v = images[:,:,0], images[:,:,1], images[:,:,2]
        v = cv2.equalizeHist(v)
        images = np.dstack((h,s,v))
        # Convert gray to the Image
        images = cv2.cvtColor(images, cv2.COLOR_HSV2BGR)
        # Define your algorithm here 👆
        return images, bboxs


class CLAHEContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        images = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
        
        h, s, v = images[:,:,0], images[:,:,1], images[:,:,2]
        #clipLimit -> Threshold for contrast limiting
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize =(8,8))
        v = clahe.apply(v)
        images = np.dstack((h,s,v))
        images = cv2.cvtColor(images, cv2.COLOR_HSV2BGR)
        # Define your algorithm here 👆
        return images, bboxs
