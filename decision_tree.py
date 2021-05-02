"""
file: decision_tree.py
language: python3
author: Chenghui Zhu    cz3348@rit.edu
description: this file contains the decision tree and its tree node class.
It also includes other formatting method for training set or testing set.
"""

import math
import pickle
import sys
from feature import *

def sample(fileName):
    """
    Formatting the example file (with language type) to training set
    :param fileName: example file
    :return: training set (each entry with boolean value in front and
    language type at the last position)
    """
    trainingSet = []
    with open(fileName, encoding="utf8") as file:
        for line in file:
            line = line.strip()
            entry = convert(line)
            trainingSet.append(recognize(entry))
    return trainingSet


def testSample(fileName):
    """
    Formating the example file (without language type) to testing set
    :param fileName: example file
    :return: testing set (each entry with only boolean value)
    """
    testSet = []
    with open(fileName, encoding="utf8") as file:
        for line in file:
            line = line.strip()
            entry = format(line)
            testSet.append(findFeature(entry))
    return testSet


def entropy(x):
    """
    Calculating the entropy value of a given possibility
    :param x: possibility input
    :return: entropy value
    """
    if x == 0 or x == 1:
        return 0
    else:
        return -(x * math.log(x, 2) + (1 - x) * math.log((1 - x), 2))


def findAttribute(list):
    """
    Summarizing each possible feature from a given training set
    :param list: the specific training set
    :return: a dictionary of {each feature: result and remainder}
    """
    attribute_remainder = {}
    attribute = len(list[0]) - 1
    total = len(list)
    for i in range(attribute):
        trueEN = 0
        trueNL = 0
        falseEN = 0
        falseNL = 0
        for j in range(total):
            if list[j][i] is True and list[j][-1] == "en":
                trueEN += 1
            elif list[j][i] is True and list[j][-1] == "nl":
                trueNL += 1
            elif list[j][i] is False and list[j][-1] == "en":
                falseEN += 1
            elif list[j][i] is False and list[j][-1] == "nl":
                falseNL += 1
        if trueEN + trueNL != 0:
            pTure = trueEN / (trueEN + trueNL)
        else:
            pTure = 0
        if falseEN + falseNL != 0:
            pFalse = falseEN / (falseEN + falseNL)
        else:
            pFalse = 0
        attribute_remainder[i] = {"trueEN": trueEN, "trueNL": trueNL,
                                  "falseEN": falseEN, "falseNL": falseNL,
                                  "remainder": entropy(pTure) * (trueEN + trueNL) / total + entropy(pFalse) * (falseEN + falseNL) / total}
    return attribute_remainder


def splitList(list, attribute, value):
    """
    Sublisting the training set by the given value at specific attribute (feature)
    :param list: the original training set
    :param attribute: the attribute (feature) value
    :param value: the language type
    :return: the sub training set
    """
    split = []
    for x in list:
        if x[attribute] == value:
            split.append(x)
    return split


def leastRemainder(data):
    """
    Finding the attribute (feature) value with the least remainder
    :param data: the dictionary from :func:`~findAttribute` method
    :return: the attribute (feature) value with the least remainder. If the
    current node cannot be further determined, this function return -1.
    """
    smallest = 1
    attribute = 0
    for i in data:
        temp = data[i]["remainder"]
        if temp < smallest:
            smallest = temp
            attribute = i
    for i in data:
        if i != attribute and data[i]["remainder"] == smallest:
            attribute = -1
            break
    return attribute


class TreeNode:
    """
    TreeNode class represents each step of a decision tree
    self.info: the specific training set
    self.stop: if the node is at bottom of the decision tree (leaf node)
    self.decision: the prediction given at this node
    self.hypothesis: a list of all pairs of (checked feature value, T/F)
    """
    def __init__(self, list):
        self.info = list
        self.features = findAttribute(list)
        self.nextFeature = leastRemainder(self.features) # = -1
        self.trueBranch = None
        self.falseBranch = None
        self.parent = None
        self.stop = False
        self.decision = "en"
        self.hypothesis = []
        # self.setNextFeature()

    def setTrue(self, node):
        """
        Setting the node as the true branch (left child) of the current node
        :param node: the child node
        :return: None
        """
        self.trueBranch = node
        node.parent = self
        for p in node.parent.hypothesis:
            node.hypothesis.append(p)
        node.hypothesis.append((self.nextFeature, True))
        if self.features[self.nextFeature]["trueEN"] >= self.features[self.nextFeature]["trueNL"]:
            node.decision = "en"
        else:
            node.decision = "nl"

    def setFalse(self, node):
        """
        Setting the node as the false branch (right child) of the current node
        :param node: the child node
        :return: None
        """
        self.falseBranch = node
        node.parent = self
        for p in node.parent.hypothesis:
            node.hypothesis.append(p)
        node.hypothesis.append((self.nextFeature, False))
        if self.features[self.nextFeature]["falseEN"] >= self.features[self.nextFeature]["falseNL"]:
            node.decision = "en"
        else:
            node.decision = "nl"

    def __repr__(self):
        return "Decision list:\n" + str(self.hypothesis) + "\nDecision: " + self.decision


class DecisionTree:
    """
    DecisionTree class represents the decision tree (root and leaf nodes)
    """
    def __init__(self, node):
        self.root = node
        self.leaves = []

    def inducing(self, node, depth=10):
        """
        Inducing the decision tree from a sepcific node if possible
        :param node: the start node
        :param depth: the max depth of the inducing can reach (default as 10)
        :return: None
        """
        if not node.stop:
            left = TreeNode(splitList(node.info, node.nextFeature, True))
            if node.features[node.nextFeature]["trueEN"] == 0 or \
                    node.features[node.nextFeature]["trueNL"] == 0:
                left.stop = True
            node.setTrue(left)
            if len(left.hypothesis) >= depth or left.features[node.nextFeature]["remainder"] == 1 or left.nextFeature == -1:
                left.stop = True

            right = TreeNode(splitList(node.info, node.nextFeature, False))
            if node.features[node.nextFeature]["falseEN"] == 0 or \
                    node.features[node.nextFeature]["falseNL"] == 0:
                right.stop = True
            node.setFalse(right)
            if len(right.hypothesis) >= depth or right.features[node.nextFeature]["remainder"] == 1 or right.nextFeature == -1:
                right.stop = True

            self.inducing(left)
            self.inducing(right)

    def getLeaf(self, node):
        """
        Collecting all leaf nodes
        :param node: the start node
        :return: None
        """
        if not node is None:
            if node.stop:
                self.leaves.append(node)
            else:
                self.getLeaf(node.trueBranch)
                self.getLeaf(node.falseBranch)

    def testSingle(self, test):
        """
        Testing a single entry, find the corresponding language type prediction
        :param test: a single list with only boolean value
        :return: the language type
        """
        temp = self.root
        while True:
            if temp.stop:
                break
            elif test[temp.nextFeature] is True:
                temp = temp.trueBranch
            elif test[temp.nextFeature] is False:
                temp = temp.falseBranch
        return temp.decision

    def testAll(self, list):
        """
        Testing all entries from a testing set
        :param list: the testing set
        :return: a list of all prediction
        """
        result = []
        for s in list:
            result.append(self.testSingle(s))
        return result

    def output(self, fileName):
        """
        Outputing the decision tree object into a file
        :param fileName: the output file name
        :return: None
        """
        with open(fileName, "wb") as file:
            pickle.dump(self, file)


def main():
    trainingSet = sample("size_1000.dat")
    root = TreeNode(trainingSet)
    dt = DecisionTree(root)
    dt.inducing(dt.root)
    dt.getLeaf(dt.root)
    for x in dt.leaves:
        print(x)

    # rT = root.trueBranch
    # print(rT)
    # print()
    #
    # rTT = rT.trueBranch
    # print(rTT)
    # print()
    #
    # rTTF = rTT.falseBranch
    # print(rTTF)
    # print()

    # print(trainingSet)
    # data = findAttribute(trainingSet)
    # print(data)
    # print(leastRemainder(data))

    # second = splitList(trainingSet, 1, True)
    # print(second)
    # data2 = findAttribute(second)
    # print(data2)
    # print(leastRemainder(data2))


# if __name__ == "__main__":
#     main()