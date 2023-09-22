"""
Writer: Nguyễn Đăng Khoa
"""

from ._utils import *


class Translate(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False
        self.randomFn = DistRand(range=(0, 1, 0.1))

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        transX = []
        transY = []

        for image in images:
            tranX = self.randomFn.rand() * image.shape[1]
            transX.append(tranX)
            tranY = self.randomFn.rand() * image.shape[0]
            transY.append(tranY)

        new_bboxs = []

        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox
            x += transX[i]
            y += transY[i]

            new_bboxs.append([x, y, w, h])

        # Define your algorithm here 👆
        return images, new_bboxs
