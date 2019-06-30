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


def KF(training_set,testing_set,file):
	
	classifier = nltk.NaiveBayesClassifier.train(training_set)
	file.write("Original Naive Bayes Algo accuracy percent:"+ str((nltk.classify.accuracy(classifier, testing_set))*100))

	MNB_classifier = SklearnClassifier(MultinomialNB())
	MNB_classifier.train(training_set)
	file.write("MNB_classifier accuracy percent:"+str((nltk.classify.accuracy(MNB_classifier, testing_set))*100))
	MNB_per = nltk.classify.accuracy(MNB_classifier, testing_set)*100


	BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
	BernoulliNB_classifier.train(training_set)	
	file.write("BernoulliNB_classifier accuracy percent:"+ str((nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100))
	BNB_per = nltk.classify.accuracy(BernoulliNB_classifier, testing_set)*100


	LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
	LogisticRegression_classifier.train(training_set)
	file.write("LogisticRegression_classifier accuracy percent:"+ str((nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100))
	LR_per = nltk.classify.accuracy(LogisticRegression_classifier, testing_set)*100

#SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
#SGDClassifier_classifier.train(training_set)
#print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

	SVC_classifier = SklearnClassifier(SVC())
	SVC_classifier.train(training_set)
	file.write("SVC_classifier accuracy percent:"+ str((nltk.classify.accuracy(SVC_classifier, testing_set))*100))
	SVC_per = nltk.classify.accuracy(SVC_classifier, testing_set)*100


	LinearSVC_classifier = SklearnClassifier(LinearSVC())
	LinearSVC_classifier.train(training_set)
	file.write("LinearSVC_classifier accuracy percent:"+ str((nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100))
	LSVC_per = nltk.classify.accuracy(LinearSVC_classifier, testing_set)*100
	

	NuSVC_classifier = SklearnClassifier(NuSVC())
	NuSVC_classifier.train(training_set)
	file.write("NuSVC_classifier accuracy percent:"+ str((nltk.classify.accuracy(NuSVC_classifier, testing_set))*100))
	Nu_SVC_per = nltk.classify.accuracy(NuSVC_classifier, testing_set)*100
	
	file.write("\n")
