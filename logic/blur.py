"""
Writer: Nguyễn Hoàng Hải
"""

from ._utils import *


class GaussianBlur(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        if self.frame_cutter:
            transformed_images = []
            for image in images:
                transformed_images.append(cv.GaussianBlur(image, (5, 5), sigma))
            sigma = DistRand(np.arange(5, 10), mode = 'unif').rand()[0]
        else:
            transformed_images = cv.GaussianBlur(images, (5, 5), sigmaX=0)
            
        # Define your algorithm here 👆
        return transformed_images, bboxs


class MedianBlur(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        if self.frame_cutter:
            transformed_images = []
            for image in images:
                transformed_images.append(cv.medianBlur(image, 5))
        else:
            transformed_images = cv.medianBlur(images, 5)
        
        # Define your algorithm here 👆
        return transformed_images, bboxs


class MotionBlurHorizontal(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # generating the kernel for horizontal blur
        size = 5
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        if self.frame_cutter:
            transformed_images = []
            for image in images:
                transformed_images.append(cv.filter2D(image, -1, kernel_motion_blur))
        else:
            transformed_images = cv.filter2D(images, -1, kernel_motion_blur)

        # Define your algorithm here 👆
        return transformed_images, bboxs

class MotionBlurVertical(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # generating the kernel for vertical blur
        size = 15
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        if self.frame_cutter:
            transformed_images = []
            for image in images:
                transformed_images.append(cv.filter2D(image, -1, kernel_motion_blur))
        else:
            transformed_images = cv.filter2D(images, -1, kernel_motion_blur)
            
        # Define your algorithm here 👆
        return transformed_images, bboxs
