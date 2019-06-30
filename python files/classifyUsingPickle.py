#open the downloaded tweets and classify them

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
import re

#module methods
from BOW_findFeature import find_features
from BOW_highFreq_pos_neg import buildPickle


#to test if file exists or not
from pathlib import Path #Python 3.4 offers it, in 2.7 we used to have 'pathlib2'


def classify_tweets(fileName):

	#check if the pickle file exists or not
	#"naivebayes_mod.pickle"
	pickle_file = Path("./naivebayes_mod.pickle")
	if pickle_file.exists():
		#print("file exists")
		pass
	else:
		buildPickle()
	
	
	# reading text     
	short_tweet = open(fileName,"r").read()#,encoding='utf8'

	#encoding
	short_tweet = short_tweet.encode('ascii','ignore').decode("utf-8")

	#data cleaning step-3 : removing Emoji --> this can be applied on text

	emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
							u"\U0001F300-\U0001F5FF"
							u"\U0001F680-\U0001F6FF"
							u"\U0001F1E0-\U0001F1FF"
							"]+",flags=re.UNICODE)
	short_tweet = emoji_pattern.sub(r'',short_tweet)


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
	

#	for slang in dict_spot:
#		if slang in short_tweet:
#			short_tweet = short_tweet.replace(slang, dict_spot[slang])
        
#striping

	short_tweet = short_tweet.strip()
	short_tweet =short_tweet.lower()

	documents = []

	for r in short_tweet.split('\n'):
		#documents.append( (r, "pos") )
		documents.append(r)
#print(len(documents))# positive tweets =5331 






	featureSets = [find_features(rev) for rev in documents]

# positive data example:      
	testing_set =  featureSets



# .pickle file: now we will read it into memory, (as quick as reading any document)
	classifier_f = open("naivebayes_mod.pickle","rb")
	classifier = pickle.load(classifier_f)
	classifier_f.close()


	y_pred = []

	count_Pos = 0
	count_Neg = 0
	for tups in testing_set:
		y_pred.append(classifier.classify(tups))
	
	for rev in y_pred:
		if rev == 'neg':
			count_Neg = count_Neg + 1
		else:
			count_Pos = count_Pos + 1
			
	print("Queries retrieved = "+ str(count_Pos + count_Neg))
	print("count_Pos = "+str(count_Pos))
	print("count_Neg = "+str(count_Neg))