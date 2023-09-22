"""
Writer: Dao Ngoc Huy
"""

import random
import albumentations as A
from ._utils import *
import math


class RandomRainTonyD(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        slant_lower = random.randint(-20, 0)
        slant_upper = random.randint(0, 20)
        random.seed(42)
        transform = A.Compose([A.RandomRain(rain_type='heavy', brightness_coefficient=0.7, blur_value=2, slant_lower=slant_lower, slant_upper=slant_upper, p=0.5)],)
        images = transform(image=images)['image']
        # Define your algorithm here 👆
        return images, bboxs

class RandomSnowTonyD(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        # random from 0-1 with step 0.1
        # snow_point_lower = random.randint(0, 10) / 10
        snow_point_upper = random.randint(5, 10) / 10
        snow_point_lower = random.randint(0, 5) / 10
        random.seed(42)
        transform = A.Compose([A.RandomSnow(brightness_coeff=2.5, snow_point_upper = snow_point_upper, snow_point_lower=snow_point_lower, p=1)],)
        images = transform(image=images)['image']
        # Define your algorithm here 👆
        return images, bboxs

class RandomSunFlareTonyD(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        src_radius = random.randint(150, 200)
        lower = random.randint(0, 8)/10
        upper = random.randint(0, 10)/10
        if lower > upper:
            lower, upper = upper, lower
        x_min = random.randint(30, 70)/100
        y_min = random.randint(0, 40)/100
        x_max = random.randint(30, 70)/100
        y_max = random.randint(0, 40)/100
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        if x_min == x_max:
            x_max += 0.1
        if y_min == y_max:
            y_max += 0.1
        random_flare_roi = (x_min, y_min, x_max, y_max)
        # random_flare_roi = (0, 0, 1, 0.5)

        print(random_flare_roi)
        random.seed(42)
        transform = A.Compose([A.RandomSunFlare(flare_roi = random_flare_roi,angle_lower = lower, angle_upper = upper,src_radius=src_radius)],)
        images = transform(image=images)['image']
        # Define your algorithm here 👆
        return images, bboxs
    
class RandomShadowTonyD(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = True

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # convert images to numpy array
        result = []
        for image in images:
            x_min = random.randint(30, 90)/100
            y_min = random.randint(30, 90)/100
            x_max = random.randint(0, 90)/100
            y_max = random.randint(0, 90)/100
            if x_min > x_max:
                x_min, x_max = x_max, x_min
            if y_min > y_max:
                y_min, y_max = y_max, y_min
            if x_min == x_max:
                x_max += 0.1
            if y_min == y_max:
                y_max += 0.1
            random_roi = (x_min, y_min, x_max, y_max)
            shadow_dimension = random.randint(3, 8)
            image = np.array(image)
            transform = A.Compose([A.RandomShadow(shadow_roi=random_roi, shadow_dimension=shadow_dimension)],)
            random.seed(42)
            image = transform(image=image)['image']
            result.append(image)
        # Define your algorithm here 👆
        return result, bboxs