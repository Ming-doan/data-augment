"""
Writer: Võ Phi Trường
"""

from ._utils import *


class SigmoidContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # Define your algorithm here 👆
        return images, bboxs


class GammarContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # Define your algorithm here 👆
        return images, bboxs


class HistogramContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # Define your algorithm here 👆
        return images, bboxs


class CLAHEContrast(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # Define your algorithm here 👆
        return images, bboxs
