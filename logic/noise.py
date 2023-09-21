"""
Writer: Nguyễn Hoàng Hải
"""

from ._utils import *


class GaussianNoise(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇
        
        # Define the mean and standard deviation of the Gaussian noise
        mean = 0
        stddev = 50  # Adjust this value to control the noise level

        if self.frame_cutter:
            noisy_images = []
            for image in images:
                # Generate Gaussian noise with the same size as the image
                noise = np.zeros_like(images, dtype=np.uint8)
                cv.randn(noise, mean, stddev)

                # Add the noise to the image
                noisy_images.append(cv.add(image, noise))
        else:
            # Generate Gaussian noise with the same size as the image
            noise = np.zeros_like(images, dtype=np.uint8)
            cv.randn(noise, mean, stddev)

            # Add the noise to the image
            noisy_images = cv.add(images, noise)

        # Define your algorithm here 👆
        return noisy_images, bboxs


class SaltAndPepperNoise(Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_cutter = False

    def transform(self, images, bboxs, width, height):
        # Define your algorithm here 👇

        # Define the probability of salt and pepper noise
        salt_prob = 0.01  # Probability of a pixel becoming white (salt)
        pepper_prob = 0.01  # Probability of a pixel becoming black (pepper)

        if self.frame_cutter:
            noisy_images = []
            for image in images:
                noisy_image = np.copy(image)

                # Generate random values for noise
                salt_mask = np.random.rand(*image.shape[:2]) < salt_prob
                pepper_mask = np.random.rand(*image.shape[:2]) < pepper_prob

                # Add salt noise
                noisy_image[salt_mask] = 255

                # Add pepper noise
                noisy_image[pepper_mask] = 0

                # Append into noisy_images list
                noisy_images.append(noisy_image)
        else:
            noisy_images = np.copy(images)

            # Generate random values for noise
            salt_mask = np.random.rand(*images.shape[:2]) < salt_prob
            pepper_mask = np.random.rand(*images.shape[:2]) < pepper_prob

            # Add salt noise
            noisy_images[salt_mask] = 255

            # Add pepper noise
            noisy_images[pepper_mask] = 0

        # Define your algorithm here 👆
        return noisy_images, bboxs
