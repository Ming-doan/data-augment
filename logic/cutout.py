"""
Writer: Nguyễn Đăng Khoa
"""

from ._utils import *


class CutOut(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # Define your algorithm here 👆
        return images, bboxs
