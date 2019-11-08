import cv2
import numpy as np
import glob
from sklearn.cluster import KMeans
import pickle
import pandas as pd
import random


class BovwKnn:

    @staticmethod
    def euclidian(vec1, vec2):
        """
        Function to compute Euclidian distance between two vectors

        :param vec1: Vector 1 - generally feature descriptor of test image
        :param vec2: Vector 2 - feature descriptors of training images
        :return: distance: Euclidian distance
        """

        distance = 0

        for i in range(len(vec1)):
            distance += (vec1[i] - vec2[i]) ** 2

        distance = np.sqrt(distance)
        return distance

    def __init__(self, training_path, validation_path, n_clusters=100, model_path=''):
        """
        Takes in the training data and creates a dictionary of images in each category
        Assumes that there are separate folders containing the images for each class

        :param training_path: Path of the training directory
        :param validation_path: Path of the validation directly
        :param n_clusters: Number of clusters to be formed
        :param model_path: Path of a pre-trained model
        """

        train_categories = glob.glob(training_path + '/*')
        val_categories = glob.glob(validation_path + '/*')

        # dictionary of training images
        self.train_dict = {}
        # dictionary of validation images
        self.val_dict = {}

        # load model if a pre-trained model is provided
        if model_path != '':
            self.kmeans = pickle.load(open(model_path, "rb"))
        else:
            # initialize new model
            self.kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)

        # store image arrays in the dictionaries
        for i in range(len(train_categories)):
            # for each category

            # name of category
            train_category = train_categories[i][train_categories[i].rfind('/') + 1:]
            val_category = val_categories[i][val_categories[i].rfind('/') + 1:]

            self.train_dict[train_category] = []
            self.val_dict[val_category] = []

            train_image_paths = glob.glob(train_categories[i] + '/*')
            val_image_paths = glob.glob(val_categories[i] + '/*')

            for img1 in train_image_paths:
                # for each training image
                self.train_dict[train_category].append(cv2.imread(img1))

            for img2 in val_image_paths:
                # for each validation image
                self.val_dict[val_category].append(cv2.imread(img2))

    def cluster(self, proportion=1):
        """
        Function to cluster the training images into classes using the kmeans model initialized in the init function

        :param proportion: Proportion of training images from each category to use
        """

        sift = cv2.xfeatures2d.SIFT_create()  # object for SIFT
        training_data = []  # contains the feature descriptors of all training images

        for key, images in self.train_dict.items():
            # for each category
            print("Category : " + key)

            for i in range(round(len(images) * proportion)):
                # for each image in the category

                (_, descriptors) = sift.detectAndCompute(images[i], None)  # obtain descriptors
                descriptors = list(descriptors)
                training_data += descriptors

        training_data = np.array(training_data, dtype=np.float)

        # fit kmeans model
        print("Fitting.....")
        self.kmeans.fit(training_data)

    def convert_to_bovw(self, data):
        """
        Function to convert each image into a bag of visual words

        :param data: The data to be converted
        :return: Dataframe containing BOVW vector and category of each image in the data
        """

        sift = cv2.xfeatures2d.SIFT_create()  # object for SIFT
        vw = []  # list of BOVW vectors for each image
        categories = []  # list of categories of all images

        for category, images in data.items():
            # for each category
            print("Category : " + category)

            for image in images:
                # for each image

                (_, descriptors) = sift.detectAndCompute(image, None)  # compute feature descriptors
                prediction = self.kmeans.predict(descriptors)  # predict the classes of the descriptors from kmeans model
                pred_histogram, _ = np.histogram(prediction, bins=self.kmeans.n_clusters)  # construct a histogram of visual words
                pred_histogram = pred_histogram / pred_histogram.sum()  # normalize histogram
                vw.append(pred_histogram)
                categories.append(category)

        bovw = pd.DataFrame(data={'Histogram': vw, 'Category': categories})  # create dataframe
        return bovw

    def knn(self, train, test, k=5):
        """
        Function to use KNN to classify the validation images

        :param train: Dataframe containing BOVW vectors and categories of training images
        :param test: Dataframe containing BOVW vectors and categories of validation images
        :param k: Number of data points that vote
        :return: confusion_matrix: The confusion matrix to visualize the classification results
        :return: success_percent: Proportion of validation images classified accurately
        """

        category_list = ['Street', 'Suburb', 'TallBuilding', 'Mountain', 'OpenCountry', 'Kitchen',
                         'Highway', 'Forest', 'Office', 'Coast']
        confusion_matrix = pd.DataFrame(0, index=category_list, columns=category_list)
        correct_predictions = 0  # number of correct predictions
        lst = []

        for index1, row1 in test.iterrows():
            # for each test image

            actual_category = row1['Category']
            distances = []  # list to store distance to each of the training images
            print('Row' + str(index1))

            for index2, row2 in train.iterrows():
                # for each training image
                distances.append(self.euclidian(row1['Histogram'], row2['Histogram']))  # calculate Euclidian distance

            train['Distance'] = distances  # add distances list to training dataframe
            sorted_train = train.sort_values(by='Distance', inplace=False, ascending=True)  # sort training dataframe based on distances

            neighbours = list(sorted_train['Category'][:k])  # extract the nearest neighbours
            predicted_category = max(set(neighbours), key=neighbours.count)  # get majority vote from nearest neighbours
            lst.append([actual_category,predicted_category])

            # check if prediction is correct
            if predicted_category == actual_category:
                correct_predictions += 1

            # update confusion matrix
            confusion_matrix.loc[actual_category, predicted_category] += 1

        # calculate prediction percentage
        success_percent = correct_predictions / len(test)

        # normalize each row to a sum of 1
        confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=1), axis=0)

        return confusion_matrix, success_percent

