"""
file: feature.py
language: python3
author: Chenghui Zhu    cz3348@rit.edu
description: This file contains all decision features of whether a sentence
(converted into a list of single words) is Dutch (True) or English (False)
"""

def convert(sentence):
    """
    Formatting a sentence to a list of words only (for training set)
    """
    trainingSet = []
    try:
        sentence = sentence.lower()
        temp = sentence.split("|")
        result = temp[0]
        words = temp[1].split(" ")
        for i in range(len(words)):
            for letter in words[i]:
                if not letter.isalpha():
                    words[i] = words[i].replace(letter, "")
        trainingSet.append(result)
        trainingSet.append(words)
    except:
        print("Invalid input. Check input sentence format.")
    return trainingSet


def has_een(list):
    # = a/an
    if "een" in list:
        return True
    else:
        return False


def has_de(list):
    # = the
    if "de" in list:
        return True
    else:
        return False


def has_bij(list):
    # = with/at
    if "bij" in list:
        return True
    else:
        return False


def has_van(list):
    # = from
    if "van" in list:
        return True
    else:
        return False


def has_conj(list):
    # = conjunctions
    conj = ["maar", "en", "als", "dan"]
    for x in conj:
        if x in list:
            return True
    return False


def has_ij(list):
    # = Personal pronouns
    ij = ["ik", "jij", "u", "gij", "hij", "zij", "wij", "het"]
    for x in ij:
        if x in list:
            return True
    return False


def has_adverb(list):
    adverb = ["this", "that", "there", "which", "where", "who", "whose", "when"]
    for x in adverb:
        if x in list:
            return False
    return True


def has_prep(list):
    prep = ["in", "on", "at", "to", "for", "by", "of", "with", "and", "or"] # "in" in both lang
    for x in prep:
        if x in list:
            return False
    return True


def has_pron(list):
    pron = ["i", "you", "he", "she", "it", "we", "they", "him", "her", "us", "them", "the", "a", "an"]
    for x in pron:
        if x in list:
            return False
    return True


def has_be(list):
    be = ["am", "is", "are", "was", "were", "being", "been", "be"]
    for x in be:
        if x in list:
            return False
    return True


def recognize(sentence):
    """
    recognize each feature in order
    :param list: single word list
    :return: a list of 10 boolean from previous features and its language type
    """
    lang = sentence[0]
    list = sentence[1]
    result = []
    result.append(has_een(list))
    result.append(has_de(list))
    result.append(has_bij(list))
    result.append(has_van(list))
    result.append(has_conj(list))
    result.append(has_ij(list))
    result.append(has_adverb(list))
    result.append(has_prep(list))
    result.append(has_pron(list))
    result.append(has_be(list))
    result.append(lang)
    return result


def format(sentence):
    """
    Formatting a sentence to a list of words only (for testing set)
    """
    sentence = sentence.lower().strip()
    words = sentence.split(" ")
    for i in range(len(words)):
        for letter in words[i]:
            if not letter.isalpha():
                words[i] = words[i].replace(letter, "")
    return words


def findFeature(list):
    """
    recognize each feature in order
    :param list: single word list
    :return: a list of 10 boolean from previous features
    """
    result = []
    result.append(has_een(list))
    result.append(has_de(list))
    result.append(has_bij(list))
    result.append(has_van(list))
    result.append(has_conj(list))
    result.append(has_ij(list))
    result.append(has_adverb(list))
    result.append(has_prep(list))
    result.append(has_pron(list))
    result.append(has_be(list))
    return result


def main():
    test = "en|Mr President, there is a German proverb that says good things are worth waiting for."
    example = convert(test)
    print(example)
    print(recognize(example))

    test2 = "Mr President, there is a German proverb that says good things are worth waiting for."
    example2 = format(test2)
    print(example2)
    print(findFeature(example2))


# if __name__ == "__main__":
#     main()