import json
from configParser import choose_function
from getArguments import spaCy, pos_tagged, check_regex


"""
Description: by using a dictionary that includes words often used 
in arguments, it is identified if given sentences are 
arguments or not. Part of speech tagging from spaCy is used 
for this purpose as well and it is imported by getArguments.py file.
"""


def check_dictionary(doc, dictionary):
    """
    This function checks if any of the words in the sentence exists
     in the dictionary given. If it does, then it is checked if
     this word's part of speech match with its value given in
     dictionary. If they match, then the word is added in a
     list named keyword_found.

    :param doc: pos tagged sentence from spacy function
    :param dictionary: dictionary that has as keywords words
                        and as value their part of speech
    :return: keywords found in the given sentence
    """

    keyword_found = []

    for key, value in dictionary.items():
        if check_regex(doc, key) is not None:
            if len(value) == 1:
                for key2 in value:
                    if check_regex(doc, key + ' ' + key2) is not None \
                            and check_regex(doc, key2) is not None \
                            and pos_tagged(doc, check_regex(doc, key2).text) is not 'None':
                        if value[key2] == pos_tagged(doc, check_regex(doc, key2).text)[1] or \
                                value[key2] == pos_tagged(doc, check_regex(doc, key2).text)[2]:
                            keyword_found.append(key + " + " + key2)
            elif value == 'none':
                keyword_found.append(key)
            elif len(value) != 1 and check_regex(doc, value) is not None:
                keyword_found.append(key)
            elif pos_tagged(doc, key)[1] == value or pos_tagged(doc, key)[2] == value:
                keyword_found.append(key)

    return keyword_found


if __name__ == '__main__':

    with open('../dict/dictionary.json', 'r') as dict:
        dictionary = json.load(dict)

    sentences = choose_function("found_ambiguous.csv")

    if sentences != 'No dataset found':
        for sentence in sentences:
            print('"' + str(sentence[0]).strip('b') + '",' +
                  str(check_dictionary(spaCy(sentence[0]), dictionary)))
    else:
        print(sentences)
