import json
import textract
import os
import csv

"""
    Description: extract words which are often used in arguments (based
                 on a paper),and create a dictionary based on these words
                 (key of the dict) and their specific, if they have one, part
                 of speech (value of the dict) in arguments
"""

FILE_PATH = os.path.abspath(os.path.dirname(__file__))  # path of this file


def extract_data():
    """
    By using textract library, this function extracts the whole pdf file

    pdf: paper called 'Using Linguistic Phenomena to Motivate a Set
                       of Coherence Relations'
    """

    text = textract.process(os.path.join(FILE_PATH,
                            "../Reading/cues-UsingLinguisticPhenomenaMotivateCoherenceRelations_Knott93.pdf"))
    save(text)


def save(text):
    """
        This function saves extracted text to a csv file
    """
    if not os.path.exists("../dict" ):
        os.makedirs("../dict")

    with open(os.path.join(FILE_PATH, "../dict/data.csv"),
              mode='wb') as csv_file:
        csv_file.write(text)

    modify_csv_file("../dict/data.csv")  # modify extracted text


def modify_csv_file(data):
    """
        This function modifies csv file in order to keep those words we are
        interested in
    """

    flag = 0

    with open(os.path.join(FILE_PATH, data)) as inp:
        reader = csv.reader(inp)

        with open(os.path.join(FILE_PATH, "../dict/data2.csv"),
                  mode='w') as out:
            for row in reader:
                if len(row) > 0 and row[0] == "Phrase":
                    flag = 1
                    continue
                if len(row) == 0 or row[0].isdigit():
                    flag = 0
                if flag == 1 and len(row) > 0:
                    out.write(row[0])
                    out.write("\n")
    check_words("../dict/data2.csv", "../dict/data.csv")


def check_words(data2, data):
    """
        This function adds or removes words that considered as useful or not
    """

    exclude_words = ['after', 'and', 'as soon as', 'before', 'at first',
                     'at first sight', 'earlier', 'fisrt of all',
                     'for', 'inasmuch as', 'later', 'much sooner', 'not because',
                     'now', 'if not', 'if so', 'in the beginning', 'in the end',
                     'in the meantime', 'in turn', 'much later', 'not',
                     'notwithstanding that', 'suppose', 'the more often',
                     'this time', 'presumably because', 'when', 'where',
                     'previously', 'regardless of that', 'rather', 'after that', 'as',
                     'simply because', 'then', 'true', 'until', 'again', 'and/or', 'or', 'else', 'even']

    include_words = ['for the reason that', 'besides', '(E|e)(ither).+?(or)',
                     '(N|n)(either).+?(nor)', 'in one hand', 'in this case',
                     'on one side', 'as a matter of fact', 'in point of fact',
                     'presumably', 'provided that', 'regardless', 'rather than',
                     'simply', 'as an example', 'in addition']

    test_words = {'even though': 'none', 'first': 'adv', 'against': 'none', 'last': 'adv',
                  'more': {'[a-z]*ly': 'adv'}, 'most': {'[a-z]*ly': 'adv'}, 'if': 'none',
                  '(T|t)(he more).+?(the more)': 'none', '(T|t)(he more).+?(the less)': 'none', 'naturally': 'none',
                  'once again': 'none', 'once more': 'none', 'surely': 'none',
                  'second': 'adv', 'so': 'mark', 'third': 'adv', 'too': '(too)($|[\.])', 'should say': 'none',
                  'might say': 'none', 'may say': 'none', 'could say': 'none',
                  'while': 'mark', 'as a start': 'none', 'in order to': 'none',
                  'in order that': 'none', 'still': 'adv', 'that is': 'none',
                  'since': 'mark', 'yet': '(Y|y)(et)[^\.].', 'that': 'mark'}

    with open(os.path.join(FILE_PATH, data2), 'r') as inp, \
            open(os.path.join(FILE_PATH, data), 'w') as out:

        for row in csv.reader(inp):
            if row[0] in exclude_words:
                continue
            else:
                out.write(row[0])
                out.write("\n")

        for word in include_words:
            out.write(word)
            out.write("\n")

    create_dictionary("../dict/data.csv", test_words)


def create_dictionary(data, test_words):
    """
        This function creates a .json file that includes a dictionary of the
        words from the csv file created before and some additional words for
        testing
    """

    dictionary = test_words

    with open(os.path.join(FILE_PATH, data), 'r') as inp:

        for row in csv.reader(inp):
            if "\x05" in row[0]:
                row[0] = row[0].replace('\x05', 'fi')  # correct words from pdf extraction

            if row[0] in test_words.keys():
                continue
            else:
                dictionary.update({row[0]: 'none'})

    with open('../dict/dictionary.json', 'w') as dict:
        json.dump(dictionary, dict)


if __name__ == '__main__':
    extract_data()
