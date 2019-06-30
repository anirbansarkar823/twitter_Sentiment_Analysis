
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
dict_spot = {'ama':'ask me anything','bc':'because','b/c':'because','b4':'because',
             'bae':'before anyone else','bd':'big deal',
	'bf':'boyfriend','bff':'best friends forever','brb':'i will be back soon','btw':'by the way',
    'cu':'see you','cyl':'see you later',
	'dftba':'do not forget to be awesome','dm':'direct message', 
    'eei5':'explain like i am 5 years old','fb':'facebook','fb':'facebook',
	'fomo':'fear of missing out','ftfy':'fixed this for you','ftw':'for the win',
    'futab':'feet up,take a break','fya':'for your amusement',
	'fye':'for your entertainment','fyi':'for your information','gtg':'got to go',
    'g2g':'got to go','gf':'girlfriend','gr8':'great',
	'gtr':'got to run','hbd':'happy birthday','ht':'hat tip','hth':'here to help',
    'ianad':'i am not a doctor',
	'ianal':'i am not a lawyer','icymi':'in case you missed it','idc':'i dont care',
    'idk':'i don\'t know','ig':'instagram',
	'iirc':'if i remember correctly','ikr':'i knonw right ?','imo':'in my opinion',
    'imho':'in my honest opinion','irl':'in real life',
	'jk':'just kidding','l8':'late','lmao':'let me know','lol':'laughing out load',
    'mcm':'mam crush monday','myob':'mind your own business',
	'mtfbwy':'may the force be with you','nbd':'no big deal','nm':'not much',
    'nsfw':'not safe for work','nts':'note to self','nvm':'nevermind','oh':'overheard',
    'omg':'oh my god','omw':'on my way','ootd':'outfit of the day','orly':'oh really',
	'pda':'public display of affection','potd':'photo of the day',
    'potus':'president of the united states','pm':'private message','ppl':'people',
    'q':'question','qq':'quick question','qotd':'quote of the day',
	'rofl':'rolling on the floor laughing','roflmao':'rolling on the floor laughing my ass off',
    'rt':'retweet','sfw':'safe for work','smh':'shaking my head',
    'tbh':'to be honest','tbt':'throwback thursday',
	'tgif':'thank god its friday','thx':'thanks','til':'too much information',
    'tmi':'too much information','ttyi':'talk to you later','ttyn':'talk to you never',
    'ttys':'talk to you soon','txt':'text','w':'with','wbu':'what about you',
	'wcw':'women crush wednesday','wdymbt':'what do you mean by that','wom':'word of mouth',
    'wotd':'word of the day','yolo':'you only live once','yt':'youtube','yw':'you are welcome',
    'awsm':'awesome','omg':'oh my god','nah':'no','asap':'as soon as possible','cya':'see you',
    'faq':'frequently asked questions','wtf':'what the fuck','u':'you','fam':'family',
    'tfw':'that feeling when','sus':'suspicious', 'k':'okay','ok':'okay','woat':'worst of all time',
    'jomo':'joy of missing out', 'fomo':'fear of missing out','suh':'what\'s up','irl':'in real life',
    'cray':'crazy','u':'you','ur':'your','gud':'good','n8':'night','n':'night','luv':'love',
    'tnx':'thanks','sm':'some','sm1':'someone','r':'are','plz':'please','peeps':'people',
    'nth':'nothing', 'btw':'by the way','hlw':'hello','bro':'brother', 'f2t':'free to talk',
    'ditto':'same here'}
	

for slang in dict_spot:
    if slang in short_neg:
        short_neg = short_neg.replace(slang, dict_spot[slang])
        
for slang in dict_spot:
    if slang in short_neg:
        short_pos = short_pos.replace(slang, dict_spot[slang])
        




all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

# data cleaning step-1: normalisation to lower
for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'
    




from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
all_words_t = nltk.pos_tag(all_words)
for i in range(len(all_words_t)):
    all_words[i] = lemmatizer.lemmatize(all_words[i],pos=get_wordnet_pos(all_words_t[i][1]))


#trying to perform lemmatization in sentence level, already achieved: n-gram level.
#striping
short_neg = short_neg.strip()
short_pos = short_pos.strip()
		
documents = []

for r in short_pos.split('\n'):
    sent = word_tokenize(r)
    sent_t = nltk.pos_tag(sent)
    for i in range(len(sent_t)):
        sent[i] = lemmatizer.lemmatize(sent[i],pos=get_wordnet_pos(sent_t[i][1]))
    r = ' '.join(sent)
    documents.append( (r, "pos") )
#print(len(documents))# positive tweets =5331 

for r in short_neg.split('\n'):
    sent = word_tokenize(r)
    sent_t = nltk.pos_tag(sent)
    for i in range(len(sent_t)):
        sent[i] = lemmatizer.lemmatize(sent[i],pos=get_wordnet_pos(sent_t[i][1]))
    r = ' '.join(sent)
    documents.append( (r, "neg") )
    
    
    
    
#data cleaning step-2: removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
all_words = [w for w in all_words if w not in stop_words]
#print(len(all_words));# 180212


all_words = nltk.FreqDist(all_words)
# returns a dictionary with words along with their respective frequency in decreasing order.

word_features = list(all_words.keys())[:8000]
#print(len(word_features))#20336
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
training_set = featureSets[0:5000] + featureSets[5662:]
testing_set =  featureSets[5000:5662]

# training the classifier as
classifier = nltk.NaiveBayesClassifier.train(training_set)


# pickle module to serialize our classifier object, so that we can just load that file in real time
#save_classifier = open("naivebayes_mod.pickle","wb")# opens a pickle file, preparing to write some bytes.
#pickle.dump(classifier, save_classifier)# 1st param: what we are dumping, 2nd param: where to dump
#save_classifier.close()
# now we have serialized file for our classifier object


# .pickle file: now we will read it into memory, (as quick as reading any document)
#classifier_f = open("naivebayes_mod.pickle","rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()



#print(testing_set)
#print("Original Naive Bayes Algo accuracy percent:", (classifier.classify(testing_set)))
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

y_pred = []
y_true = []

for tups in testing_set:
	y_pred.append(classifier.classify(tups[0]))
	y_true.append(tups[1])


# using confusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np

#COPIED FUNCTION
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
	
	
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    print("accuracy=",accuracy)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()



tn, fp, fn, tp = confusion_matrix(y_true,y_pred,labels=["pos","neg"]).ravel()
print(tn,fp,fn,tp)
plot_confusion_matrix(np.array(confusion_matrix(y_true,y_pred,labels=["pos","neg"])),normalize = False, target_names = ['pos','neg'],title="confusion_matrix")
#nltkNB_per = nltk.classify.accuracy(classifier, testing_set)*100




