import cv2
import numpy as np
import math


class Filters:
    """
    A class for implementing convolution, correlation and median filtering

    Attributes
    ----------
    image: numpy matrix
        a numpy matrix that represents the image
    channels: int
        number of channels in the image

    Methods
    ----------
    convolution(kernel)
        function to implement convolution operation on image
    correlation(kernel)
        function to implement correlation operation on image
    median(kernel_shape)
        function to implement median filtering on an image
    dot_product(kernel, kernel_mid_index)
        computes the dot_product between the kernel and the image window
    """

    def __init__(self, image_path):
        """
        :param image_path: Path of the image
        """

        self.image = cv2.imread(image_path)  # reading the image in OpenCV - format is BGR

        # code to initialize number of channels
        self.channels = 0
        if len(self.image.shape) == 3:
            self.channels = self.image.shape[-1]
        else:
            self.channels = 1

    def dot_product(self, kernel, kernel_mid_index):
        """
        Computes the dot_product between the kernel and the image window
        
        :param kernel: Numpy matrix representing the kernel 
        :param kernel_mid_index: Index of the mid element of the kernel
        :return: A filtered image
        """
        filtered_image = np.zeros(shape=self.image.shape, dtype=int)

        # loop through each channel
        for channel in range(self.channels):

            # loop through each image pixel
            for i in range(0, self.image.shape[0]):
                for j in range(0, self.image.shape[1]):

                    modified_pixel = 0

                    # loop through each kernel pixel
                    for u in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                        for v in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                            if (i + u) >= 0 and (j + v) >= 0:
                                # correlation operation
                                try:
                                    modified_pixel += kernel[kernel_mid_index - u, kernel_mid_index - v] * self.image[
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

    def correlation(self, kernel):
        """
        Function to implement correlation operation on image
        
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
        filtered_image = self.dot_product(kernel, kernel_mid_index)  # apply filter operation
        return filtered_image

    def convolution(self, kernel):
        """
        Function to implement convolution operation on image
        
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

        filtered_image = self.dot_product(kernel, kernel_mid_index)  # apply filter operation
        return filtered_image

    def median(self, kernel_shape):
        """
        Function to implement median filtering on an image
        
        :param kernel_shape: Tuple representing shape of the kernel 
        :return: A filtered image
        """

        # checks for kernel shape correctness
        if len(kernel_shape) != 2:
            print("Kernel should be 2D\n")
            exit(-1)
        if kernel_shape[0] != kernel_shape[1]:
            print("Kernel should be square")
            exit(-1)
        if (kernel_shape[0] - 1) % 2 != 0:  # derived from kernel size = (2k+1, 2k+1)
            print("Kernel should have a middle point")
            exit(-1)

        filtered_image = np.zeros(shape=self.image.shape, dtype=int)
        kernel_mid_index = int((kernel_shape[0] - 1) / 2)  # derived from kernel size = (2k+1, 2k+1)

        # loop through each channel
        for channel in range(self.channels):

            # loop through each image pixel
            for i in range(0, self.image.shape[0]):
                for j in range(0, self.image.shape[1]):

                    considered_pixels = []

                    # loop through each kernel pixel
                    for u in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                        for v in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                            if (i + u) >= 0 and (j + v) >= 0:
                                try:
                                    # list to calculate the median
                                    considered_pixels.append(self.image[i + u, j + v, channel])
                                except IndexError:
                                    considered_pixels.append(0)  # emulates the effect of zero padding
                            else:
                                considered_pixels.append(0)

                    # calculate the median
                    modified_pixel = np.median(considered_pixels)

                    if len(filtered_image.shape) == 3:
                        filtered_image[i, j, channel] = modified_pixel
                    else:
                        filtered_image[i, j] = modified_pixel

        return filtered_image

    @staticmethod
    def gaussian_kernel(shape, sigma):
        """Function to generate 2D Gaussian Kernel given shape and sigma value"""
        
        gaussian_kernel = np.zeros(shape=shape)

        kernel_mid_index = int((shape[0] - 1) / 2)  # derived from kernel size = (2k+1, 2k+1)

        for i in range(-kernel_mid_index, kernel_mid_index + 1, 1):
            for j in range(-kernel_mid_index, kernel_mid_index + 1, 1):
                # 2D Gaussian function implementation
                gaussian_kernel[i + kernel_mid_index, j + kernel_mid_index] = (1 / (2 * np.pi * sigma * sigma)) * \
                                                                              (1 / math.exp(((i * i) + (j * j)) /
                                                                                            (2 * sigma * sigma)))

        return gaussian_kernel/gaussian_kernel.sum()  # normalizing
