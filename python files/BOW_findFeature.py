# -*- coding: utf-8 -*-

# Imporving Training Data for sentiment analysis with NLTK
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

from nltk.tokenize import word_tokenize
import re


# reading text     
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

#encoding
short_neg = short_neg.encode('ascii','ignore').decode("utf-8")# to encode

short_pos = short_pos.encode('ascii','ignore').decode("utf-8")
#short_neg = short_neg.strip()#strip off edge spaces
#short_neg = re.sub(r'\s+',' ',short_neg,flags=re.I)# strip off within spaces which are present in more than one

#data cleaning step-4 : split attached words--> will not work here as all words are in lower()
#short_neg_temp = re.findall('[A-Z][^A-Z]*',short_neg)
#print(short_neg_temp)
#short_neg = ' '.join(short_neg_temp)

#data cleaning step-3 : removing Emoji --> this can be applied on text

emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
							u"\U0001F300-\U0001F5FF"
							u"\U0001F680-\U0001F6FF"
							u"\U0001F1E0-\U0001F1FF"
							"]+",flags=re.UNICODE)
short_pos = emoji_pattern.sub(r'',short_pos)

short_neg = emoji_pattern.sub(r'',short_neg)
#print(short_neg)

#standardising words
#import itertools
#short_neg = ''.join(''.join(s)[:2] for _,s in itertools.groupby(short_neg))
#short_pos = ''.join(''.join(s)[:2] for _,s in itertools.groupby(short_pos))


# step-12: Removal of slang
# slang words and their corresponding meaning.
dict_spot = {'ama':'ask me anything','bc':'because','b/c':'because','b4':'because','bae':'before anyone else','bd':'big deal',
	'bf':'boyfriend','bff':'best friends forever','brb':'i will be back soon','btw':'by the way','cu':'see you','cyl':'see you later',
	'dftba':'do not forget to be awesome','dm':'direct message', 'eei5':'explain like i am 5 years old','fb':'facebook','fb':'facebook',
	'fomo':'fear of missing out','ftfy':'fixed this for you','ftw':'for the win','futab':'feet up,take a break','fya':'for your amusement',
	'fye':'for your entertainment','fyi':'for your information','gtg':'got to go','g2g':'got to go','gf':'girlfriend','gr8':'great',
	'gtr':'got to run','hbd':'happy birthday','ht':'hat tip','hth':'here to help','ianad':'i am not a doctor',
	'ianal':'i am not a lawyer','icymi':'in case you missed it','idc':'i dont care','idk':'i dont know','ig':'instagram',
	'iirc':'if i remember correctly','ikr':'i knonw right ?','imo':'in my opinion','imho':'in my honest opinion','irl':'in real life',
	'jk':'just kidding','l8':'late','lmao':'let me know','lol':'laughing out load','mcm':'mam crush monday','myob':'mind your own business',
	'mtfbwy':'may the force be with you','nbd':'no big deal','nm':'not much','nsfw':'not safe for work','nts':'note to self','nvm':'nevermind','oh':'overheard','omg':'oh my god','omw':'on my way','ootd':'outfit of the day','orly':'oh really',
	'pda':'public display of affection','potd':'photo of the day','potus':'president of the united states','pm':'private message','ppl':'people','q':'question','qq':'quick question','qotd':'quote of the day',
	'rofl':'rolling on the floor laughing','roflmao':'rolling on the floor laughing my ass off','rt':'retweet','sfw':'safe for work','sm':'social media','smh':'shaking my head','tbh':'to be honest','tbt':'throwback thursday',
	'tgif':'thank god its friday','thx':'thanks','til':'too much information','tmi':'too much information','ttyi':'talk to you later','ttyn':'talk to you never','ttys':'talk to you soon','txt':'text','w':'with','wbu':'what about you',
	'wcw':'women crush wednesday','wdymbt':'what do you mean by that','wom':'word of mouth','wotd':'word of the day','yolo':'you only live once','yt':'youtube','yw':'youre welcome'}
	


#striping
short_neg = short_neg.strip()
short_pos = short_pos.strip()
#short_pos =short_pos.lower()
#short_neg = short_neg.lower()
documents = []

for r in short_pos.split('\n'):
    documents.append( (r, "pos") )
#print(len(documents))# positive tweets =5331 

for r in short_neg.split('\n'):
    documents.append( (r, "neg") )
#print(len(documents)) 5331



all_words_pos = []
all_words_neg = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

# data cleaning step-1: normalisation to lower
for w in short_pos_words:
    all_words_pos.append(w.lower())

for w in short_neg_words:
    all_words_neg.append(w.lower())
	
#removing slang
for iteri,slang in enumerate(all_words_neg):
    if slang in dict_spot:
        all_words_neg[iteri] = dict_spot[slang]
        
for iteri,slang in enumerate(all_words_pos):
    if slang in dict_spot:
        all_words_pos[iteri] = dict_spot[slang]

#data cleaning step-2: removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
all_words_pos = [w for w in all_words_pos if w not in stop_words]
all_words_neg = [w for w in all_words_neg if w not in stop_words]
#print(len(all_words));# 153244


all_words_pos = nltk.FreqDist(all_words_pos)
all_words_neg = nltk.FreqDist(all_words_neg)

# returns a dictionary with words along with their respective frequency in decreasing order.

word_features_pos = list(all_words_pos.keys())[:5000]
word_features_neg = list(all_words_neg.keys())[:5000]
word_features = word_features_pos + word_features_neg

#print(len(word_features))
def find_features(tweet):
    words_of_tweet = word_tokenize(tweet)
    features = {}
	
	# for each word out of the all 5000 most appeared words
    for w in word_features:
        features[w] = (w in words_of_tweet)# (w in words) returns true or false based on its presence inside the current tweet
    return features

featureSets = [(find_features(rev), category) for (rev, category) in documents]
#print(len(featureSets))#10662 --> total tweets
random.shuffle(featureSets)

# positive data example:      
training_set = featureSets
	#testing_set =  featureSets[5000:5662]

# training the classifier as
classifier = nltk.NaiveBayesClassifier.train(training_set)


# pickle module to serialize our classifier object, so that we can just load that file in real time
save_classifier = open("naivebayes_mod.pickle","wb")# opens a pickle file, preparing to write some bytes.
pickle.dump(classifier, save_classifier)# 1st param: what we are dumping, 2nd param: where to dump
save_classifier.close()
# now we have serialized file for our classifier object


