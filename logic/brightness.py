from ._utils import *


class Brightness(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rand = DistRand((0.5, 1.5))

    def transform(self, images, bboxs, width, height):
        images = images * self.rand.rand()

        return images, bboxs
