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