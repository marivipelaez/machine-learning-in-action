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
from __future__ import division
import matplotlib.pyplot as plt


DECISION_NODE = dict(boxstyle='sawtooth', fc='0.8')
LEAF_NODE = dict(boxstyle='round4', fc='0.8')
ARROWS_SETTINGS = dict(arrowstyle='<-')


def plot_basic_node(node_text, center_pt, parent_pt, node_type):
    """Define a node in the plot with based on its type configuration
    :param node_text:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    """

    create_plot.ax1.annotate(node_text, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction', va='center', ha='center',
                             bbox=node_type, arrowprops=ARROWS_SETTINGS)


def create_plot():
    """Basic function to plot a decision tree. It uses a hard coded tree.
    """

    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_basic_node('decision node', (0.5, 0.1), (0.1, 0.5), DECISION_NODE)
    plot_basic_node('leaf node', (0.8, 0.1), (0.3, 0.8), LEAF_NODE)
    plt.show()


def get_number_of_leaves(tree):
    """Loops the tree recursively to find out its number of leafs. For plotting purposes, it is the width of the
    tree plot.

    :param tree: dict of dicts object with the structure of a decision tree

        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

    :return: integer number of leafs in the given tree
    """

    num_leafs = 0
    first_label = tree.keys()[0]
    second_dict = tree[first_label]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            num_leafs += get_number_of_leaves(second_dict[key])
        else:
            num_leafs += 1

    return num_leafs


def get_tree_depth(tree):
    """Loops the tree recursively to find out the depth of the tree. For plotting purposes, it is the height of the
    tree plot.

    :param tree: dict of dicts object with the structure of a decision tree

        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}

     :return: integer depth of the given tree
    """

    max_depth = 0
    first_label = tree.keys()[0]
    second_dict = tree[first_label]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            current_depth = 1 + get_tree_depth(second_dict[key])
        else:
            current_depth = 1
        if current_depth > max_depth:
            max_depth = current_depth

    return max_depth


def retrieve_tree(tree):
    """Example of trees for testing purposes

        >>> import ch3_trees.trees_plotter as trees_plotter
        >>> trees_plotter.retrieve_tree(0)
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        >>> tree_0 = trees_plotter.retrieve_tree(0)
        >>> trees_plotter.get_number_of_leafs(tree_0)
        3
        >>> trees_plotter.get_tree_depth(tree_0)
        2

    :param tree: integer with the identifier of the tree to be plotted
    :returns: the selected tree in the form of nested dictionaries
    """
    trees = [
        {'no surfacing': {0: 'no',
                          1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return trees[tree]


def plot_node(node_text, center_pt, parent_pt, node_type):
    """Define a node in the plot with based on its type configuration
    :param node_text:
    :param center_pt:
    :param parent_pt:
    :param node_type:
    """

    create_tree_plot.ax1.annotate(node_text, xy=parent_pt, xycoords='axes fraction',
                                  xytext=center_pt, textcoords='axes fraction', va='center', ha='center',
                                  bbox=node_type, arrowprops=ARROWS_SETTINGS)


def plot_mid_text(child_node, parent_node, message):
    """ Utility function that calculates the mid point between child_point and parent_point and
    prints the message there.

    :param child_node: tuple of float numbers with the (x, y) coordinates of the child node
    :param parent_node: tuple of float numbers with the (x, y) coordinates of the parent node
    :param message: string to be printed in the connection between a child and a parent node.
    :returns:
    """

    # As __future__ is imported integer division is float
    mid_x = (parent_node[0] - child_node[0])/2 + child_node[0]
    mid_y = (parent_node[1] - child_node[1])/2 + child_node[1]

    create_tree_plot.ax1.text(mid_x, mid_y, message)


def plot_tree(tree, parent_node, node_text):
    """Plots a tree, hanging on parent_point. This function is called recursively to plot every nested tree.

    :param tree: dictionary to be plotted
    :param parent_node: node where the tree hangs on.
    :param node_text: message in the connection between the parent node and the new tree
    :returns:
    """

    # Get the width and the height of a tree
    num_leaves = get_number_of_leaves(tree)
    depth = get_tree_depth(tree)

    # Get first key in tree
    key = tree.keys()[0]
    child_node = (plot_tree.xoff + (1 + num_leaves)/2/plot_tree.total_width, plot_tree.yoff)

    plot_mid_text(child_node, parent_node, node_text)
    plot_node(key, child_node, parent_node, DECISION_NODE)
    next_tree = tree[key]
    plot_tree.yoff += - 1/plot_tree.total_depth

    for key in next_tree.keys():
        if isinstance(next_tree[key], dict):
            plot_tree(next_tree[key], child_node, str(key))
        else:
            plot_tree.xoff += 1/plot_tree.total_width
            plot_node(next_tree[key], (plot_tree.xoff, plot_tree.yoff), child_node, LEAF_NODE)
            plot_mid_text((plot_tree.xoff, plot_tree.yoff), child_node, str(key))

    plot_tree.yoff += 1/plot_tree.total_depth


def create_tree_plot(tree):
    """Prepares a plot panel to show the figure with the decision tree.

    :param tree: nested dictionaries structure containing the decision tree.
    :returns: Plots the tree
    """

    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_tree_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # Configure plot_tree function variables
    plot_tree.total_width = get_number_of_leaves(tree)
    plot_tree.total_depth = get_tree_depth(tree)
    plot_tree.xoff = -.5/plot_tree.total_width
    plot_tree.yoff = 1.0
    starting_point = (.5, 1.0)
    # The first time the whole tree is sent, with the selected starting point and without a text.
    plot_tree(tree, starting_point, '')
    plt.show()
