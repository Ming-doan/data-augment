from ._utils import *


class Squeeze(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = True
        self.rand = DistRand((1, 0.5), mode='expo')

    def transform(self, images, bboxs, width, height):

        probs = [self.rand.rand() for _ in range(len(images))]

        new_imgs = []
        for i, image in enumerate(images):
            _placholder = create_image_placeholder(image, 'right-pixels')
            image = cv.resize(
                image, (int(image.shape[1] * probs[i]), image.shape[0]))
            _placholder[:, :image.shape[1], :] = image
            new_imgs.append(_placholder)

        new_bboxs = []
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox
            new_bboxs.append((x, y, w * probs[i], h))

        return new_imgs, new_bboxs
