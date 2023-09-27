"""
Writer: Nguyễn Hoàng Hải
"""

from ._utils import *
from sklearn.cluster import MeanShift, estimate_bandwidth


class MeanShiftSegmentation(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        def mean_shift(image):
            # Flatten the image to a list of RGB pixels
            pixels = image.reshape((-1, 3))
            pixels = np.float32(pixels)

            # Estimate the bandwidth (you can adjust this value)
            bandwidth = estimate_bandwidth(
                pixels, quantile=0.06, n_samples=3000)

            # Perform Mean Shift clustering
            ms_clustering = MeanShift(
                bandwidth=bandwidth, max_iter=800, bin_seeding=True)
            ms_clustering.fit(pixels)

            # Get the labels and cluster centers
            labels = ms_clustering.labels_
            cluster_centers = ms_clustering.cluster_centers_

            # Create a segmented image using cluster labels
            segmented_image = cluster_centers[labels].reshape(
                image.shape).astype(np.uint8)

            return segmented_image

        if self.frame_cutter:
            segmented_images = []
            for image in images:
                segmented_images.append(mean_shift(image))
        else:
            segmented_images = mean_shift(images)

        # Define your algorithm here 👆
        return segmented_images, bboxs
