"""
Writer: Đoàn Quang Minh
"""

from ._utils import *


def rotate(pos: tuple[int], deg: float, image: Image = None, values: tuple[float] = None):
    # Define translate to center matrix
    translate_to_center = np.array([
        [1, 0, -pos[0]],
        [0, 1, -pos[1]],
        [0, 0, 1]
    ])
    # Define rotate matrix
    rotate = np.array([
        [np.cos(deg), -np.sin(deg), 0],
        [np.sin(deg), np.cos(deg), 0],
        [0, 0, 1]
    ])
    # Define translate to origin matrix
    translate_to_origin = np.array([
        [1, 0, pos[0]],
        [0, 1, pos[1]],
        [0, 0, 1]
    ])
    # Define transform matrix
    transform = translate_to_origin @ rotate @ translate_to_center
    # Apply transform
    if image is not None:
        # Apply transform to image
        return cv.warpAffine(image, transform[:-1], image.shape[:2][::-1])
    elif values is not None:
        # Apply transform to values
        return (transform @ np.array([[*values, 1]]).T).T[0][:-1]


class Rotation(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get mode of rotation
        self.mode = kwargs.get("mode", 'all')  # single, all
        self.frame_cutter = True if self.mode == 'single' else False
        # Define random
        boundary = kwargs.get("boundary", 20) * np.pi / 180
        self.random = DistRand(range=(-boundary, boundary))

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        if self.mode == 'single':
            new_images = []
            # Only rotate images
            for image in images:
                angle = self.random.rand()
                # Get image shape
                h, w, _ = image.shape
                image = rotate((w // 2, h // 2), angle, image=image)
                # Append to new images
                new_images.append(image)
            # Update images
            images = new_images

        elif self.mode == 'all':
            angle = self.random.rand()
            images = rotate((width // 2, height // 2), angle, image=images)
            # Update bboxs
            new_bboxs = []
            for bbox in bboxs:
                x, y, w, h = bbox
                # Get center of bbox
                cx, cy = x + w / 2, y + h / 2
                # Update bbox
                new_poss = rotate((width // 2, height // 2),
                                  angle, values=(cx, cy))
                cx, cy = new_poss
                # Update bbox
                x, y = cx - w / 2, cy - h / 2
                # Append to new bboxs
                new_bboxs.append((x, y, w, h))
            # Update bboxs
            bboxs = new_bboxs

        else:
            assert False, "Mode of rotation is not valid. Please choose 'single' or 'all'"

        # Define your algorithm here 👆
        return images, bboxs
