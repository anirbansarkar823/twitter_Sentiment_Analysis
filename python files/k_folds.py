import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

import pandas
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import k_folds_f





documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
 
x = [(find_features(rev)) for (rev, category) in documents]
y =  [(category) for (rev, category) in documents]
random.shuffle(featuresets)
 
skf = StratifiedKFold(n_splits=10)

    # blank lists to store predicted values and actual values
predicted_y = []
expected_y = []

    # partition data
training_set = []
testing_set = []
file = open("output_kfold.txt","w")
file.close

for train_index, test_index in skf.split(x, y):
	file = open("output_kfold.txt","a")
	file.write(str(train_index)+ str(test_index))
        # specific ".loc" syntax for working with dataframes
   # x_train, x_test = x[train_index], x[test_index]
    #y_train, y_test = y[train_index], y[test_index]
	for i in train_index:
		training_set.append(featuresets[i])
	for i in test_index:
		testing_set.append(featuresets[i])
		
	k_folds_f.KF(training_set,testing_set,file)   
    #accuracy = metrics.accuracy_score(expected_y, predicted_y)
    #print("Accuracy: " + accuracy.__str__())
	#print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)
	#nltkNB_per = nltk.classify.accuracy(classifier, testing_set)*100
	file.close()	