# Argument Mining

_This project is the assignment of 'Final Year Project' undergraduate course of Department of Management Science and Technology at Athens University of Economics and Business_
## Definition
Argument mining is a relatively new research field in natural language processing. The purpose of argument mining is to understand what kind of views have been expressed in the examined text and why they are held.

## Reasearch Goal
My research goal is to identify argumentative statements by using two different approaches;
the structural approach which is based on hand coded rules and the statistical approach,
which based on supervised and deep learning algorithms.

The fundamental research questions that will be addressed in this assignment are the
following:
- To what extent are the lexical rules drafted by a structural approach capable of suc-
cessfully identifying arguments in existing resources of labeled data?
- Do the statistical approaches outperform these results?

## Chapters
1. Introduction
2. State-of-the-Art
3. Methods
4. Data Curation
5. Results
6. Conclusion

## Installation
In order to run the source code execute the following steps:

1. Clone the repository into your local machine.

`git clone https://github.com/kfragkedaki/ArgumentMining`

2. Make sure you have installed Python 3.6 or newer version. If not, then install it by using the following commands:

`sudo apt-get install software-properties-common`
`sudo add-apt-repository ppa:deadsnakes/ppa`
`sudo apt-get update`
`sudo apt-get install python3.7 `

3. Install the required Python packages by executing the shell script **install_packages.sh** which is available in the repo.

`sh install_packages.sh`

4. Download [http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip] and add the `glove.6B.100d.txt` in the new directory named *glove.6B* inside *Reading* folder. After the procedure, the file should be in this relative path `../Reading/glove.6B/glove.6B.100d.txt`

#### Supervisors
Professor Panos Louridas, Senior Researcher Vasiliki Efstathiou
#### Annotators
Senior Researcher Vasiliki Efstathiou, Student Klio Fragkedaki

