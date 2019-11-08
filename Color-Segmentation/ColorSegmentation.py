import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 12.8, 9.6  # size of scatter plot


class ColorSegmentation:

    def __init__(self, path):
        """
        Function to read training images in HSV and store max Hue and Separation values

        :param path: Path of the folder containing training images
        """

        self.images = []
        self.maxH = 0
        self.maxS = 0

        # paths of all training images
        image_files = glob.glob(path)

        for file in image_files:

            img = cv2.imread(file)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to HSV
            self.images.append(img_hsv)

            if img_hsv[:, :, 0].max() > self.maxH:
                self.maxH = img_hsv[:, :, 0].max()  # record maxH

            if img_hsv[:, :, 1].max() > self.maxS:
                self.maxS = img_hsv[:, :, 1].max()  # record maxS

        self.histogram = np.zeros((self.maxH + 1, self.maxS + 1), dtype=np.uint8)  # initialize empty array for histogram

    def create_histogram(self, plot):
        """
        Creates a frequency histogram to record occurrences of different (H, S) pairs and plots a scatter plot

        :param plot: Name of the file to store scatter plot
        :return: self.histogram: Normalized histogram of H-S frequencies
        """

        # for every training image
        for image in self.images:

            for row in range(0, image.shape[0]):
                for col in range(0, image.shape[1]):
                    # looping through each pixel

                    h = image[row, col, 0]
                    s = image[row, col, 1]
                    self.histogram[h, s] += 1  # increment frequency counter

        self.histogram = self.histogram / self.histogram.max()  # normalize histogram

        # plot and save scatter plot
        x, y = np.meshgrid(range(self.maxH + 1), range(self.maxS + 1))
        x = np.reshape(x, -1)
        y = np.reshape(y, -1)
        z = np.reshape(self.histogram, -1)
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c=z, cmap='Accent', linewidth=1)
        plt.xlabel("Hue")
        plt.ylabel("Saturation")
        plt.savefig(plot)

    def extract_skin(self, image_path, dest_path, proportion):
        """
        Function to extract skin regions from input images

        :param image_path: Path of input image
        :param dest_path: Path to store the result
        :param proportion: Proportion of maximum frequency to be used as lower threshold
        """

        image_RGB = cv2.imread(image_path)
        image_HSV = cv2.cvtColor(image_RGB, cv2.COLOR_BGR2HSV)  # convert input image to HSV
        threshold = proportion * self.histogram.max()  # find threshold frequency

        new_image = np.zeros((image_RGB.shape[0], image_RGB.shape[1], 3))

        all_H = np.where(self.histogram > threshold)[0]  # indexes of all hue values that are above threshold
        all_S = np.where(self.histogram > threshold)[1]  # indexes of all saturation values that are above threshold
        all_HS = []

        for i in range(len(all_H)):

            # get (H, S) neighbourhood
            all_HS.append((all_H[i], all_S[i]))
            all_HS.append((all_H[i] - 1, all_S[i] + 1))
            all_HS.append((all_H[i] + 1, all_S[i] - 1))
            all_HS.append((all_H[i] - 1, all_S[i] - 1))
            all_HS.append((all_H[i] + 1, all_S[i] + 1))
            all_HS.append((all_H[i], all_S[i] - 1))
            all_HS.append((all_H[i], all_S[i] + 1))
            all_HS.append((all_H[i] - 1, all_S[i]))
            all_HS.append((all_H[i] + 1, all_S[i]))
        all_HS = list(set(all_HS))

        for row in range(image_HSV.shape[0]):
            for col in range(image_HSV.shape[1]):
                # for every pixel in input image

                HS_value = (image_HSV[row, col, 0], image_HSV[row, col, 1])

                # check if H and S value of pixel is above threshold
                if HS_value in all_HS:
                    new_image[row, col, :] = image_RGB[row, col, :]  # qualify pixel
                else:
                    new_image[row, col, :] = image_RGB[row, col, :] * 0  # disqualify pixel

        cv2.imwrite(dest_path, new_image)  # write result





