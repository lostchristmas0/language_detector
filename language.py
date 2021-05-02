"""
file: language.py
language: python3
author: Chenghui Zhu    cz3348@rit.edu
description: This file contains the main function of this project.
"""

import math
import sys
import pickle
from feature import *
from decision_tree import *
from adaboost import *

DT_DEPTH = 10
SAMPLE_ARGUMENTS = "train size_100.dat dtout1 dt", "train size_100.dat adaout1 ada", \
                   "predict dtout1 test1.dat", "predict adaout1 test1.dat"
INFO = "Language Classification ver1.0\n-------------------------------------------------------------------------\n" \
       "Function syntax:\n(1) train *training example file* *learning object output file* learning option(dt/ada)\n" \
       "taking the example file as a training set and learning as either decision tree or ada boost\n\n" \
       "(2) predict *learning object input file* *testing example file*\n" \
       "predicting language type in the example file via previous learning object\n\n" \
       "(3) help\nshowing help message\n\nSample argument:"


def train(trainingSet, hypothesisOut, learningType):
    """
    The train function
    :param trainingSet: input training set
    :param hypothesisOut: the output file name
    :param learningType: the learning type (whether dt or ada)
    :return: None
    """
    if learningType == "dt":
        root = TreeNode(trainingSet)
        dt = DecisionTree(root)
        dt.inducing(dt.root, DT_DEPTH)
        dt.getLeaf(dt.root)
        dt.output(hypothesisOut)
    elif learningType == "ada":
        ada = AdaBoost()
        ada.training(trainingSet, ada.hypothesis)
        ada.output(hypothesisOut)
    else:
        raise IOError("invalid input learning type. (must be 'dt' or 'ada')")


def predict(hypothesis, file):
    """
    The predict function
    :param hypothesis: the input learning object
    :param file: the testing example file
    :return: the corresponding prediction list
    """
    testSet = testSample(file)
    return hypothesis.testAll(testSet)


def inputfile(fileName):
    """
    Inputing a file and transforming it into its original class
    :param fileName: input file name
    :return: the learning object
    """
    with open(fileName, "rb") as file:
        data = pickle.load(file)
        return data


def main():
    if sys.argv[1].lower() == "train":
        trainingSet = sample(sys.argv[2])
        hypothesisOut = sys.argv[3]
        learningType = sys.argv[4].lower()
        train(trainingSet, hypothesisOut, learningType)
        fx = lambda x: "decision tree" if x == "dt" else ("ada boost" if x == "ada" else "???")
        print("Training data: '" + sys.argv[2] + "' with " +
              fx(learningType) + ". Object saved as: '" + sys.argv[3] + "'")
    elif sys.argv[1].lower() == "predict":
        hypothesis = inputfile(sys.argv[2])
        testFile = sys.argv[3]
        prediction = predict(hypothesis, testFile)
        for x in prediction:
            print(x)
    elif sys.argv[1].lower() == "help":
        print(INFO)
        print(SAMPLE_ARGUMENTS)
    else:
        raise IOError("invalid input entry. (available function: train/predict/help)")


if __name__ == "__main__":
    main()