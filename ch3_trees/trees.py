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