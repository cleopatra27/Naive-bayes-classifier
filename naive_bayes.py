import glob
import os
import re
from collections import defaultdict
from math import log
from tabulate import tabulate

class naive_bayes():

    def __init__(self):
        self.ham_emails = glob.glob("HamSpam/ham/*")
        self.test_emails = glob.glob("HamSpam/test/*")
        self.alpha = 0.0005
        self.spam_emails = glob.glob("HamSpam/spam/*")
        self.spam_dict = {}
        self.ham_dict = {}
        self.spam_prior = 0
        self.ham_prior = 0
        self.spam_count = 0
        self.ham_count = 0
        self.vocabulary = 0

    def populate_dict(self, files):
        word_dict = []
        word_count = 0
        for filename in files:
            with open(os.path.join(os.getcwd(), filename), 'r') as f_name:
                line = f_name.read().split('\n')
                for sentence in line:
                    # remove numbers
                    words = sentence.split()
                    for word in words:
                        word = re.sub(r'[^a-zA-Z]', "", word)
                        if word != "":
                            word_count += 1
                            word_dict.append(word.lower())
        return word_dict, word_count

    def cal_prior_prob(self):
        total_count = len(self.spam_emails) + len(self.ham_emails)
        return log(len(self.spam_emails) / total_count), log(len(self.ham_emails) / total_count)

    def likelihood(self, word_list, files):
        word_frequency = defaultdict(lambda: 0)  # counter
        temp_dict = {}
        for word in word_list:
            if word in word_list:
                word_frequency[word] += 1
            likelihood = (word_frequency[word] + self.alpha) / (len(files) + self.alpha * self.vocabulary)
            temp_dict[word.lower()] = likelihood
        return temp_dict

    def predict(self, text):
        words = text.split()
        self.spam_prior, self.ham_prior = self.cal_prior_prob()
        for word in words:
            word = word.lower()
            if word in self.spam_dict:  # check if word is in dictionary for seen word
                self.spam_prior += log(self.spam_dict[word])
            else:  # add for not seen word
                self.spam_prior += log(self.alpha / (self.spam_count + self.alpha * self.vocabulary))

            if word in self.ham_dict:  # check if word is in dictionary for seen word
                self.ham_prior += log(self.ham_dict[word])
            else:  # add for not seen word
                self.ham_prior += log(self.alpha / (self.ham_count + self.alpha * self.vocabulary))

        if self.spam_prior >= self.ham_prior:
            return "spam"
        else:
            return "ham"

    def test(self):
        spam_temp = self.populate_dict(self.spam_emails)
        ham_temp = self.populate_dict(self.ham_emails)

        self.vocabulary = len(list(dict.fromkeys(spam_temp[0]))) + len(list(dict.fromkeys(ham_temp[0])))
        self.spam_dict, self.spam_count = self.likelihood(spam_temp[0], self.spam_emails), spam_temp[1]
        self.ham_dict, self.ham_count = self.likelihood(ham_temp[0], self.ham_emails), ham_temp[
            1]

        self.predict_test_set()

        # print(self.predict("Good day, awaiting your response on the budget"))

    def predict_test_set(self):
        with open(os.path.join(os.getcwd(), "HamSpam/truthfile"), 'r') as truth_file:
            data = truth_file.read().split('\n')
            t_file = data

        counter_not_earn = 0
        counter_earn = 0
        TN = 0
        FP = 0
        FN = 0
        TP = 0

        list = []
        for file in self.test_emails:
            with open(os.path.join(os.getcwd(), file), 'r') as f_test:
                argmax = self.predict(f_test.read())

            if file[13:].split(".words")[0].strip() in t_file:
                if argmax == "spam":
                    TP += 1
                    counter_earn += 1
                    file_delim = "TP"
                else:
                    FP += 1
                    counter_not_earn += 1
                    file_delim = "FP"
            else:
                if argmax == "ham":
                    TN += 1
                    counter_earn += 1
                    file_delim = "TN"
                else:
                    FN += 1
                    file_delim = "FN"
                    counter_not_earn += 1
            list.append([file_delim, argmax, file[13:].split(".words")[0].strip()])

        Precision = TP / (TP + FP)
        Recall = (TP / (TP + FN))
        FScore = (2 * Precision * Recall) / (Precision + Recall)

        print("Precision => ", Precision)
        print("Recall => ", Recall)
        print("FScore => ", FScore)

        print('TRUTH FILE TABLE')
        print('----------------')
        print(tabulate(list, headers=['values', 'prediction', 'Email number']))

        print('\nREPORT')
        print('----------------')
        print(tabulate([[TP, TN, FP, FN]], headers=['TP', 'TN', 'FP', 'FN']))

        print('\nMEASURES')
        print('----------------')
        print(tabulate([["Precision", Precision],
                        ["Recall", Recall],
                        ["FScore", FScore]], headers=['measure', 'value']))


p = naive_bayes()
p.test()
