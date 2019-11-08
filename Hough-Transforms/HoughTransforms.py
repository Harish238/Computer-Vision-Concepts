import cv2
import numpy as np


class HoughTransforms:

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

        # diagonal of the image shape derived from pythagoras theorem
        self.rho_max = np.ceil(np.sqrt(self.image.shape[0]**2 + self.image.shape[1]**2))

        self.radians_per_pixel = 0  # self-explanatory

    def generate_accumulator(self, height, width):
        """
        Function to generate an accumulator array of zeroes

        :param height: desired height of the accumulator array
        :param width: desired width of the accumulator array
        :return: accumulator_array: an array of zeroes with shape (height, width)
        """

        accumulator_array = np.zeros((height, width), dtype=np.uint8)
        self.radians_per_pixel = np.pi / width  # initialize this value here

        return accumulator_array

    def canny(self, high_threshold=50):
        """
        Function to detect edges in image using OpenCV's Canny function

        :param high_threshold: Threshold above which edges are classified as strong edges
        :return: edges: returns the edges in an image
        """

        edges = cv2.Canny(self.image, high_threshold*0.5, high_threshold)  # function for canny edge detection
        return edges  # binary image with detected edges

    def accumulate(self, edges, accumulator):
        """
        Function to accumulate votes in the rho-theta parameter space

        :param edges: Image with edges
        :param accumulator: Accumulator array
        :return: Returns an accumulated array with votes for each position
        """

        edge_indexes = np.argwhere(edges)  # list of indexes of non-zero pixels

        for index in edge_indexes:

            # for each detected edge
            x = index[1]
            y = index[0]
            theta = 0

            # for each pixel in the width of the image
            for i in range(0, accumulator.shape[1]):

                rho = x*np.cos(theta) + y*np.sin(theta)  # theta is the radians value at the ith pixel

                rho_position_percent = ((rho - (-self.rho_max)) / (self.rho_max - (-self.rho_max)))  # percentile of the rho in the range
                rho_position = int(rho_position_percent*(accumulator.shape[0]))  # final pixel position of row in accumulator
                theta_position = i  # pixel position of theta
                accumulator[rho_position if rho_position != accumulator.shape[0] else rho_position - 1,
                            theta_position] += 1  # incrementing

                theta += self.radians_per_pixel

        return accumulator

    def find_maxima(self, accumulator, threshold=5):
        """
        Function to find the local maxima in an accumulator array

        :param accumulator: Accumulator array with votes in each position
        :param threshold: Value above which to consider the rho, theta value
        :return: lines: List of lines represented by their rho and theta values
        """

        rho_indexes = np.where(accumulator >= threshold)[0]  # stores row values of all pixels above threshold
        thetas_indexes = np.where(accumulator >= threshold)[1]  # stores column values of all pixels above threshold
        lines = []

        # for each accumulator pixel above threshold
        for i in range(len(rho_indexes)):

            rho_pixel = rho_indexes[i]
            theta_pixel = thetas_indexes[i]

            # back-calculating the rho value from the pixel position
            rho_position = rho_pixel / accumulator.shape[0]
            rho = (rho_position * (self.rho_max - (-self.rho_max))) + (-self.rho_max)

            # back-calculating theta value from pixel position
            theta = theta_pixel * self.radians_per_pixel

            lines.append((rho, theta))

        # returns list of (rho, theta) pairs
        return lines

    def draw(self, lines, filename):
        """
        Function to draw the Hough lines on the image

        :param filename: Name of the file to be written to
        :param lines: Hough lines in the form of (rho, theta)
        :return: Writes the original image with Hough lines on the disk
        """

        final_image = np.copy(self.image)

        # for each line
        for line in lines:

            rho = line[0]
            theta = line[1]

            alpha = np.cos(theta)
            beta = np.sin(theta)
            x1 = int((alpha * rho) + 1000 * -beta)
            y1 = int((beta * rho) + 1000 * alpha)
            x2 = int((alpha * rho) - 1000 * -beta)
            y2 = int((beta * rho) - 1000 * alpha)

            # draw line on the image
            cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        cv2.imwrite(filename, final_image)
