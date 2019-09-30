import os

FILE_PATH = os.path.abspath(os.path.dirname(__file__))


def remove_duplicate_rows(file_name):
    sentences = []

    with open(os.path.join(FILE_PATH, '../Results/' + file_name), mode='r') as txt_file:
        reader = txt_file.read()

        for sentence in reader.split("\n"):
            sentences.append(sentence)

    with open ( os.path.join ( FILE_PATH, '../Results/new_' + file_name ), mode='w' ) as txt_file_new:
        for i in range(len(sentences)):
            if i == 0:
                txt_file_new.write(sentences[i])
                txt_file_new.write("\n")
            elif i < len(sentences) and sentences[i] != sentences[i-1]:
                txt_file_new.write(sentences[i])
                txt_file_new.write("\n")


if __name__ == '__main__':
    remove_duplicate_rows('CDEdata.csv')
