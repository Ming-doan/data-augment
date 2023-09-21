"""
Writer: Võ Phi Trường
"""

from PIL import ImageEnhance
from ._utils import *
import numpy as np
import cv2

class Brightness(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False
        self.rand = DistRand(range=(0.85, 1.15, 0.01),mode='norm')

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        
        #Generate a random brightness factor betwwen 0.85 and 1.15 step 0.01 for each image
        num = self.rand.rand()
        
        new_images = np.power(images, num)
        
        # Define your algorithm here 👆
        return new_images, bboxs
