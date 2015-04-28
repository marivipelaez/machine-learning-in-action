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

from math import log


def get_shannon_entropy(dataset):
    """Calculates the Shannon entropy of a given dataset:
        H = -sum(n, i=1)[p(xi)log2p(xi)]
    The higher the entropy, the more mixed up the data is.

    :param dataset: data to split in the format of a list of lists. All the lists are of the same length. The last
     item in the list is the class of the element.
    :returns: shannon entropy of the dataset
    """

    num_classes = {}
    for item in dataset:
        # Get the class of the item, that is in the last element of the list
        current_class = item[-1]
        if current_class not in num_classes.keys():
            num_classes[current_class] = 0

        num_classes[current_class] += 1

    shannon_entropy = 0.0
    num_entries = len(dataset)
    for clazz, num in num_classes.items():
        prob = float(num)/num_entries
        shannon_entropy -= prob * log(prob, 2)

    return shannon_entropy


def get_simple_dataset():
    """Simple dataset for testing trees

        >>> import ch3_trees.trees as trees
        >>> dataset, features = trees.get_simple_dataset()
        >>> print(get_shannon_entropy(dataset))
        >>> 0.970950594455
        >>> dataset[0][-1] = 'maybe'
        >>> print(get_shannon_entropy(dataset))
        >>> 1.37095059445

    :returns: (dataset, features)
        * the first element is a dataset, a list of list with three elements, [1, 1, 'yes'], it means that the
        animal does not go in the surface and has flippers, so it is a fish.
        * the second element, is a list with all the available features
    """
    dataset = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    features = ['no surfacing', 'flippers']

    return dataset, features


def split_dataset(dataset, axis, axis_value):
    """Split the given dataset for the feature in axis, and taken into account its value in axis_value.

        >>> import ch3_trees.trees as trees
        >>> dataset, features = trees.get_simple_dataset()
        >>> trees.split_dataset(dataset, 0, 1)
        >>> [[1, 'yes'], [1, 'yes'], [0, 'no']]
        >>> trees.split_dataset(dataset, 0, 0)
        >>> [[1, 'no'], [1, 'no']]

    :param dataset: list of lists. All the lists are of the same length. The last item in the list is the class
    of the element.
    :param axis: index of a feature in the element list
    :param axis_value: value of the selected feature
    :return: the same dataset without the already analyzed feature and only with those elements with the given value
    """
    # To not modify the original dataset, because it is a list of lists
    reduced_dataset = []
    for features in dataset:
        if features[axis] == axis_value:
            reduced_features = features[:axis]
            reduced_features.extend(features[axis+1:])
            reduced_dataset.append(reduced_features)

    return reduced_dataset


def choose_best_splitting_feature(dataset):
    """Loops recursively through the whole dataset to determine the best feature to split it

        >>> import ch3_trees.trees as trees
        >>> dataset, features = trees.get_simple_dataset()
        >>> trees.choose_best_splitting_feature(dataset)
        >>> 0

     This means that the feature in the first position (0) has the higher information gain.

    :param dataset: list of lists. All the lists are of the same length. The last item in the list is the class
    of the element. There is no assumptions on the type of the data.
    :return: index of the best splitter feature of the dataset
    """

    # Do not count the class of the element in the number of features of the dataset
    num_features = len(dataset[0]) - 1
    # Entropy of the whole dataset without any splitting, latter use in comparisons
    base_entropy = get_shannon_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        # Creates a list with all the possible values of the feature i
        features_list = [item[i] for item in dataset]
        # Removes duplicate values
        unique_values = set(features_list)
        # Contains the entropy of the splitted dataset
        current_entropy = 0.0
        for value in unique_values:
            splitted_dataset = split_dataset(dataset, i, value)
            prob = len(splitted_dataset)/float(len(dataset))
            # Sum up entropy for all the unique values of the feature in analysis
            current_entropy += prob * get_shannon_entropy(splitted_dataset)
        info_gain = base_entropy - current_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature
