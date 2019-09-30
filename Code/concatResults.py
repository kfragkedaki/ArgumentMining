import os
import glob
import csv

FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def combine_results():

    csvfiles = glob.glob(FILE_PATH + '/../Results/checked/*.csv')
    dataset = csv.writer(open(FILE_PATH + '/../Results/dataset.csv', 'w'), delimiter=',')

    for files in csvfiles:
        rd = csv.reader(open(files, 'r'))

        for row in rd:
            if 'wiki' in files:
                if "False" in row:
                    dataset.writerow(row)
            else:
                if "True" or "False" in row:
                    dataset.writerow(row)


def find_ambiguous():

    csvfiles = glob.glob(FILE_PATH + '/../Results/checked/*.csv')
    found_ambiguous = csv.writer(open(FILE_PATH + '/../Results/found_ambiguous.csv', 'w'), delimiter=',')

    for files in csvfiles:
        rd = csv.reader(open(files, 'r'))

        for row in rd:
            if 'wiki' in files:
                if "True" in row:
                    found_ambiguous.writerow(row)


if __name__ == '__main__':
    combine_results()
    find_ambiguous()
