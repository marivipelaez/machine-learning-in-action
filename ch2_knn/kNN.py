# -*- coding: utf-8 -*-

u"""Copyright 2015 Mariví Peláez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

def create_dataset():
    """Creates the already-labeled dataset
    """

    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify(test_vector, training_dataset, training_classes, k):
    """Classification for kNN algorithm, using Euclidean distance

        d = sqrt((x1 - x2)^2 + (y1 - y2)^2)

    :param test_vector: vector of elements to be classified
    :param training_dataset: matrix of training elements
    :param training_classes: sorted vector of classes of the training dataset
    :param k: number of neighbors to use in the comparison algorithm
    :return: class of the given test_vector
    """
    # shape[0] = number of rows
    # shape[1] = number of columns

    # Number of elements in the training dataset
    dataset_size = training_dataset.shape[0]

    # Distance calculation using Euclidean distance
    diff_mat = np.tile(test_vector, (dataset_size, 1)) - training_dataset
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5

    # Sorting distances
    sorted_distances_indices = distances.argsort()

    # Getting k lowest distances
    class_count = {}
    for i in range(k):
        vote_class = training_classes[sorted_distances_indices[i]]
        class_count[vote_class] = class_count.get(vote_class, 0) + 1

    # Sorting output dictionary
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]


def file_to_matrix(filename):
    """Reads a data file and converts it into a matrix
        filename format: field1, field2, field3, class

    Execute:
        >>> import ch2_knn.kNN as knn
        >>> dating_matrix, dating_labels = knn.file_to_matrix('data/ch2/dating-test-set.txt')

    :param filename: absolute path to the file containing the data
    :returns: a matrix with the training examples and a vector with its classes
    """

    with open(filename) as f:
        lines = f.readlines()
        num_lines = len(lines)
        result_mat = np.zeros((num_lines, 3))
        class_labels = []

        index = 0
        for line in lines:
            line = line.strip()
            # Fields are tab separated
            line_content = line.split('\t')
            result_mat[index, :] = line_content[0: 3]

            # Add label, that is the last field
            class_labels.append(line_content[-1])

            index += 1

        return result_mat, class_labels


def dating_scatter_plot0(dating_matrix):
    """
    Scatter plot for dating dataset, without differentiating labels

    plot y = liters of ice cream consumed per week
    plot x = % time spent playing video games

    :param dating_matrix:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_matrix[:, 1], dating_matrix[:, 2])
    plt.show()


def dating_scatter_plot(dating_mat, dating_labels):
    """Scatter plot for dating dataset, showing labels to make it easier to see the patterns

    :param dating_mat: training dataset
    :param dating_labels: clases of the training dataset
    """
    labels_num = {'didntLike': 0, 'smallDoses': 1, 'largeDoses': 2}

    dating_labels_num = [labels_num[item] for item in dating_labels]

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('% time spent playing video games')
    ax.set_ylabel('liters of ice cream consumed per week')

    # plot y = liters of ice cream consumed per week
    # plot x = % time spent playing video games
    ax.scatter(dating_mat[:, 1], dating_mat[:, 2], 15.0*np.array(dating_labels_num), 15.0*np.array(dating_labels_num))

    # plot x = frequent flier miles earned per year
    # plot y = % time spent playing video games

    ax = fig.add_subplot(2, 1, 2)
    ax.set_xlabel('frequent flier miles earned per year')
    ax.set_ylabel('% time spent playing video games')
    ax.scatter(dating_mat[:, 0], dating_mat[:, 1], 15.0*np.array(dating_labels_num), 15.0*np.array(dating_labels_num))

    plt.show()


def normalizer(dataset):
    """Normalize the values of a dataset depending on its maximum and minimum values
    :param dataset: dataset to be normalized
    :returns: normalized dataset, ranges per column, min_values per column
    """
    # dataset.min(0) => Get minimum from the columns, not from the rows
    min_values = dataset.min(0)
    max_values = dataset.max(0)

    # Get the values ranges per column
    ranges = max_values - min_values

    norm_dataset = np.zeros(np.shape(dataset))
    m = dataset.shape[0]
    # Tile creates a matrix of the same size of the input, and adds as many tiles (copies of itself)
    # as marked in the second argument. In this case: min_values.shape=(1x3), and the output is a matrix
    # of 1000x3, with 1000 copies of min_values
    norm_dataset = dataset - np.tile(min_values, (m, 1))
    norm_dataset = norm_dataset/np.tile(ranges, (m, 1))
    return norm_dataset, ranges, min_values
