import numpy as np
import cv2


class CannyEdgeDetection:

    def __init__(self, path):
        """
        :param path: Path of the image
        """

        self.image = cv2.imread(path)  # reading the image from path in OpenCV - format BGR

        # initialize number of channels
        self.n_channels = 0
        if len(self.image.shape) == 3:
            self.n_channels = self.image.shape[-1]
        else:
            self.n_channels = 1

    @staticmethod
    def convolution(image, n_channels, kernel):
        """
        Function to implement convolution operation on image

        :param image: Image to be convolved
        :param n_channels: No of channels in the image
        :param kernel: Numpy matrix representing the kernel
        :return: A filtered image
        """

        # checks for kernel shape correctness
        if len(kernel.shape) != 2:
            print("Kernel should be 2D\n")
            exit(-1)
        if kernel.shape[0] != kernel.shape[1]:
            print("Kernel should be square")
            exit(-1)
        if (kernel.shape[0] - 1) % 2 != 0:  # derived from kernel size = (2k+1, 2k+1)
            print("Kernel should have a middle point")
            exit(-1)

        kernel_mid_index = int((kernel.shape[0] - 1) / 2)  # derived from kernel size = (2k+1, 2k+1)

        # flipping the kernel
        kernel = np.flip(kernel, 0)  # along Y-axis
        kernel = np.flip(kernel, 1)  # along X-axis

        filtered_image = np.zeros(shape=image.shape, dtype=int)

        # loop through each channel
        for channel in range(n_channels):

            # loop through each image pixel
            for i in range(0, image.shape[0]):
                for j in range(0, image.shape[1]):

                    modified_pixel = 0

                    # loop through each kernel pixel
                    for u in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                        for v in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                            if (i + u) >= 0 and (j + v) >= 0:
                                # correlation operation
                                try:
                                    modified_pixel += kernel[kernel_mid_index - u, kernel_mid_index - v] * image[
                                        i + u, j + v, channel]
                                except IndexError:
                                    modified_pixel += 0  # emulates the effect of zero padding
                            else:  # if index is out of bounds
                                modified_pixel += 0  # emulates the effect of zero padding

                    # assign new pixel value
                    if len(filtered_image.shape) == 3:
                        filtered_image[i, j, channel] = modified_pixel
                    else:
                        filtered_image[i, j] = modified_pixel

        return filtered_image

    def gaussian_smoothing(self, shape, sigma):
        """
        Function to smooth an image using a Gaussian kernel

        :param shape: Shape of Gaussian kernel required
        :param sigma: Pixel width
        :return smoothed_image: Image smoothed by a Gaussian kernel
        """

        gaussian_kernel = np.zeros(shape=shape)

        kernel_mid_index = int((shape[0] - 1) / 2)  # derived from kernel size = (2k+1, 2k+1)

        for i in range(-kernel_mid_index, kernel_mid_index + 1, 1):
            for j in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                # 2D Gaussian function implementation
                gaussian_kernel[i + kernel_mid_index, j + kernel_mid_index] = (1 / (2 * np.pi * sigma * sigma)) * \
                                                                              (1 / np.exp(((i * i) + (j * j)) /
                                                                                          (2 * sigma * sigma)))

        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()  # normalizing

        smoothed_image = self.convolution(self.image, self.n_channels,
                                          gaussian_kernel)  # smooth the image with the kernel

        return smoothed_image

    def image_gradient(self, input_image):
        """
        Function to calculate gradient magnitude and gradient direction of an image

        :param input_image: Image
        :return: gradient_magnitude: Magnitude of image gradient for each pixel
        :return: gradient_direction: Direction of image gradient for each pixel
        """

        sobel_x = np.array([[1 / 8, 0, -1 / 8], [1 / 4, 0, -1 / 4], [1 / 8, 0, -1 / 8]],
                           dtype=np.float)  # sobel operator for vertical edges
        sobel_y = np.transpose(sobel_x)  # sobel operator for horizontal edges

        df_dx = self.convolution(input_image, n_channels=self.n_channels,
                                 kernel=sobel_x)  # image gradient along horizontal direction
        df_dy = self.convolution(input_image, n_channels=self.n_channels,
                                 kernel=sobel_y)  # image gradient along vertical direction

        gradient_magnitude = np.sqrt((df_dx ** 2) + (df_dy ** 2))
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255  # normalizing magnitude to the 0-255 scale
        gradient_direction = np.arctan(np.nan_to_num(df_dy / df_dx))
        gradient_direction = gradient_direction * 180 / np.pi  # converting radians to degrees for next step

        return gradient_magnitude, gradient_direction

    def non_max_suppression(self, gradient_magnitude, gradient_direction):
        """
        Function to suppress non-maxima pixels in the gradient intensity matrix

        :param gradient_magnitude: Magnitude of image gradients
        :param gradient_direction: Direction of image gradients
        :return: new_matrix: Converted matrix where every edge is one pixel wide and composed of local maximas
        """

        new_matrix = np.zeros(gradient_magnitude.shape)
        for channel in range(self.n_channels):
            for u in range(self.image.shape[0]):
                for v in range(self.image.shape[1]):
                    # loop through every pixel

                    pixel_direction = gradient_direction[u, v, channel]

                    try:
                        if -22.5 < pixel_direction <= 22.5:  # first category of directions
                            # ternary operation for comparison with the competitors
                            new_matrix[u, v, channel] = gradient_magnitude[u, v, channel] \
                                if gradient_magnitude[u, v, channel] >= gradient_magnitude[u, v + 1, channel] and \
                                   gradient_magnitude[u, v, channel] >= gradient_magnitude[u, v - 1, channel] \
                                else 0

                        elif 22.5 < pixel_direction <= 67.5:  # second category of directions
                            # ternary operation for comparison with the competitors
                            new_matrix[u, v, channel] = gradient_magnitude[u, v, channel] \
                                if gradient_magnitude[u, v, channel] >= gradient_magnitude[u - 1, v + 1, channel] and \
                                   gradient_magnitude[u, v, channel] >= gradient_magnitude[u + 1, v - 1, channel] \
                                else 0

                        elif (67.5 < pixel_direction <= 90) or (-90 <= pixel_direction <= -67.5):  # third category of directions
                            # ternary operation for comparison with the competitors
                            new_matrix[u, v, channel] = gradient_magnitude[u, v, channel] \
                                if gradient_magnitude[u, v, channel] >= gradient_magnitude[u + 1, v, channel] and \
                                   gradient_magnitude[u, v, channel] >= gradient_magnitude[u + 1, v, channel] \
                                else 0

                        elif -67.5 < pixel_direction <= -22.5:  # fourth category of directions
                            # ternary operation for comparison with the competitors
                            new_matrix[u, v, channel] = gradient_magnitude[u, v, channel] \
                                if gradient_magnitude[u, v, channel] >= gradient_magnitude[u + 1, v + 1, channel] and \
                                   gradient_magnitude[u, v, channel] >= gradient_magnitude[u - 1, v - 1, channel] \
                                else 0

                    # boundary pixels are left unchanged
                    except IndexError:
                        new_matrix[u, v, channel] = gradient_magnitude[u, v, channel]

        return new_matrix

    @staticmethod
    def threshold(input_image, high_threshold=255, low_threshold_ratio=0.5):
        """
        Function to threshold the intensities based on given values

        :param input_image: Image to be thresholded
        :param high_threshold: Upper threshold value
        :param low_threshold_ratio: Lower threshold value expressed in proportion to high threshold
        :return: thresholded_image: Thresholded image
        """

        strong_edge_value = 255  # value to be set for strong edges
        weak_edge_value = 20  # value to be set for weak edges
        non_edge_value = 0  # value to be set for non-edges

        thresholded_image = np.zeros(input_image.shape)

        low_threshold = high_threshold * low_threshold_ratio

        # boolean masks for each condition
        strong = (input_image >= high_threshold)
        weak = ((input_image >= low_threshold) & (input_image < high_threshold))
        non = (input_image < low_threshold)

        # setting the value
        thresholded_image[strong] = strong_edge_value
        thresholded_image[weak] = weak_edge_value
        thresholded_image[non] = non_edge_value

        return thresholded_image

    def edge_link(self, input_image):
        """
        Function to link strong and weak edges

        :param input_image: Image to be edge-linked
        :return: Image with edges linked
        """

        new_matrix = np.zeros(input_image.shape)

        for channel in range(self.n_channels):
            for u in range(0, input_image.shape[0]):
                for v in range(1, input_image.shape[1]):

                    # loop through every channel
                    try:
                        # for every weak pixel
                        if input_image[u, v, channel] == 20:
                            # check if strong pixel is in 8-neighbourhood
                            if 255 in [input_image[u, v + 1, channel], input_image[u, v - 1, channel],
                                       input_image[u + 1, v, channel], input_image[u - 1, v, channel],
                                       input_image[u - 1, v - 1, channel], input_image[u + 1, v + 1, channel],
                                       input_image[u - 1, v + 1, channel], input_image[u + 1, v - 1, channel]]:
                                # convert weak pixel to strong pixel
                                new_matrix[u, v, channel] = 255
                            else:
                                # discard weak pixel
                                new_matrix[u, v, channel] = 0
                        else:
                            new_matrix[u, v, channel] = input_image[u, v, channel]
                    # boundary pixels are converted to non-edges
                    except IndexError:
                        new_matrix[u, v] = 0

        return new_matrix
