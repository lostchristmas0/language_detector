"""
file: adaboost.py
language: python3
author: Chenghui Zhu    cz3348@rit.edu
description: This file contains the ada boost learning algorithm
"""

import math
import sys
import pickle
from feature import *
from decision_tree import *

def normalize(w):
    """
    The normalization function used in ada boost
    :param w: original list of weight
    :return: normalized list of weight
    """
    factor = 1 / sum(w)
    for i in range(len(w)):
        w[i] = w[i] * factor
    return w


class AdaBoost:
    """
    The AdaBoost class representing the ada boost learning algorithm
    """
    def __init__(self):
        """
        self.hypothesis contians all 10 features as its hypothesis
        self.weight is the weight of hypothesis
        """
        self.hypothesis = []
        self.hypothesis.append(self.h0)
        self.hypothesis.append(self.h1)
        self.hypothesis.append(self.h2)
        self.hypothesis.append(self.h3)
        self.hypothesis.append(self.h4)
        self.hypothesis.append(self.h5)
        self.hypothesis.append(self.h6)
        self.hypothesis.append(self.h7)
        self.hypothesis.append(self.h8)
        self.hypothesis.append(self.h9)
        # for n in nodeList:
        #     self.hypothesis.append((n.hypothesis, n.decision))
        self.weight = []
    
    def h0(self, list):
        if list[0] is True:
            return "nl"
        else:
            return "en"

    def h1(self, list):
        if list[1] is True:
            return "nl"
        else:
            return "en"

    def h2(self, list):
        if list[2] is True:
            return "nl"
        else:
            return "en"

    def h3(self, list):
        if list[3] is True:
            return "nl"
        else:
            return "en"

    def h4(self, list):
        if list[4] is True:
            return "nl"
        else:
            return "en"

    def h5(self, list):
        if list[5] is True:
            return "nl"
        else:
            return "en"

    def h6(self, list):
        if list[6] is True:
            return "nl"
        else:
            return "en"

    def h7(self, list):
        if list[7] is True:
            return "nl"
        else:
            return "en"

    def h8(self, list):
        if list[8] is True:
            return "nl"
        else:
            return "en"

    def h9(self, list):
        if list[9] is True:
            return "nl"
        else:
            return "en"

    def training(self, example, hypo):
        """
        Performing the ada boost learning algorithm and updating the weight list
        :param example: training set
        :param hypo: original hypothesis list
        :return: None
        """
        hAmount = len(hypo)
        eAmount = len(example)
        w = [1/eAmount for _ in range(eAmount)]
        z = [1 for _ in range(hAmount)]
        for k in range(hAmount):
            error = 0
            for j in range(eAmount):
                if hypo[k](example[j]) != example[j][-1]:
                    error += w[j]
            if error <= 0 or error >= 1:
                continue
            for j in range(eAmount):
                if hypo[k](example[j]) == example[j][-1]:
                    w[j] = w[j] * error / (1 - error)
            w = normalize(w)
            z[k] = math.log((1 - error) / error)
        self.weight = z

    def testSingle(self, list):
        """
        Testing a single entry, find the corresponding language type prediction
        :param test: a single list with only boolean value
        :return: the language type
        """
        hx = []
        for i in range(len(self.hypothesis)):
            if self.hypothesis[i](list) == "nl":
                hx.append(self.weight[i])
            elif self.hypothesis[i](list) == "en":
                hx.append(-self.weight[i])
        if sum(hx) >= 0:
            return "nl"
        else:
            return "en"

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
    ada = AdaBoost()
    ada.training(trainingSet, ada.hypothesis)
    print(ada.weight)

    test2 = "Mr President, there is a German proverb that says good things are worth waiting for."
    example2 = findFeature(format(test2))
    print(ada.testSingle(example2))

    test3 = "werd het dienstgebouw opgetrokken, dat zich eveneens onder een schilddak bevindt, langs de straatzijde verspringend "
    example3 = findFeature(format(test3))
    print(example3)
    print(ada.testSingle(example3))

    # trainingSet = sample("size_1000.dat")
    # root = TreeNode(trainingSet)
    # dt = DecisionTree(root)
    # dt.inducing(dt.root)
    # dt.getLeaf(dt.root)
    #
    # test = sample("size_100.dat")
    # ada = AdaBoost(dt.leaves)
    # ada.training(test, ada.hypothesis)
    # print(ada.weight)


# if __name__ == "__main__":
#     main()