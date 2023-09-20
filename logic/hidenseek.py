"""
Writer: Đào Ngọc Huy
"""

from ._utils import *
import random

class HideAndSeek(Method):
    def __init__(self, *args, **kwargs)->None:
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        # Hide and seek algorithm

        # possible grid size, 0 means no hiding
        grid_sizes = random.sample(range(0, 100, 10), 10)

        # hiding probability
        hide_prob = 0.5
        # randomly choose one grid size
        grid_size= grid_sizes[random.randint(0,len(grid_sizes)-1)]

        # hide the patches
        if(grid_size!=0):
            for x in range(0,width,grid_size):
                for y in range(0,height,grid_size):
                    x_end = min(width, x + grid_size)
                    y_end = min(height, y + grid_size)
                    if(random.random() <=  hide_prob):
                        images[x:x_end,y:y_end,:]=0


        # Define your algorithm here 👆
        return images, bboxs
