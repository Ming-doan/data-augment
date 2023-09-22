"""
Writer: Nguyễn Đăng Khoa
"""

from ._utils import *


class CutOut(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = True
        self.randomLength = DistRand(range=(1, 2, 0.1))
        self.randomColorVariable = DistRand(range=(0, 255))

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        for i, image in enumerate(images):
            x, y, w, h = bboxs[i]

            # Get cutout box coordinates
            cutX = int(w // 2 * self.randomLength.rand())
            cutY = int(h // 2 * self.randomLength.rand())
            length = int(w // 2 * self.randomLength.rand())

            # Get random color for cutout box
            rVal = int(self.randomColorVariable.rand())
            gVal = int(self.randomColorVariable.rand())
            bVal = int(self.randomColorVariable.rand())

            # Draw cutout box onto picture
            image = cv.rectangle(image, (cutX, cutY), (cutX + length, cutY + length), (rVal, gVal, bVal), -1)

        # Define your algorithm here 👆
        return images, bboxs
