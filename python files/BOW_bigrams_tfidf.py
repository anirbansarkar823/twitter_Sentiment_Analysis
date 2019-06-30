

import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.tokenize import word_tokenize
import re
from sklearn.metrics import confusion_matrix
#texts = [
#    "good movie", "not a good movie", "did not like", 
#    "i like it", "good one"
#]
# using default tokenizer in TfidfVectorizer
short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

#encoding
short_neg = short_neg.encode('ascii','ignore').decode("utf-8")# to encode

short_pos = short_pos.encode('ascii','ignore').decode("utf-8")


#data cleaning step-3 : removing Emoji --> this can be applied on text

emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"
							u"\U0001F300-\U0001F5FF"
							u"\U0001F680-\U0001F6FF"
							u"\U0001F1E0-\U0001F1FF"
							"]+",flags=re.UNICODE)
short_pos = emoji_pattern.sub(r'',short_pos)

short_neg = emoji_pattern.sub(r'',short_neg)




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
    if slang in short_pos:
        short_pos = short_pos.replace(slang, dict_spot[slang])
        
#striping
short_neg = short_neg.strip()
short_pos = short_pos.strip()

#converting them to list of tweets
document = []
labels = []

for r in short_pos.split('\n'):
    document.append((r,1))
	#lables.append(1)
#print(len(documents))# positive tweets =5331 

for r in short_neg.split('\n'):
    document.append((r,0))
	#labels.append(0)
#print(len(documents))# positive tweets =5331
#file1 = open("tfidf_results.txt","w")

vectorizer = TfidfVectorizer(min_df=5, max_df=0.8,sublinear_tf = True,
use_idf=True,stop_words = 'english', ngram_range=(1, 2))

random.shuffle(document)
training_set = document[662:]# + document[9662:]
testing_set =  document[:662]

X_train = []
X_test = []

Y_train = []
Y_test = []
for tweet in training_set:
	X_train.append(tweet[0])
	Y_train.append(tweet[1])

for tweet in testing_set:
	X_test.append(tweet[0])
	Y_test.append(tweet[1])

train_corpus_tf_idf = vectorizer.fit_transform(X_train)
test_corpus_tf_idf = vectorizer.transform(X_test)


from sklearn.svm import SVC, LinearSVC, NuSVC
classifier = NuSVC()
classifier.fit(train_corpus_tf_idf,Y_train)

result = classifier.predict(test_corpus_tf_idf)
print(confusion_matrix(Y_test, result))
#print(result)

#features = tfidf.fit_transform(document)
#ile1.write(str(pd.DataFrame(
  #  features.todense(),
  #  columns=tfidf.get_feature_names()
#)))
#file1.close()




# WE NEED TO PLOT IT IN CONFUSION MATRIX

y_pred = []
for elem in result:
	if elem == 1:
		y_pred.append('pos')
	else:
		y_pred.append('neg')


y_true = []

for elem in Y_test:
	if elem == 1:
		y_true.append('pos')
	else:
		y_true.append('neg')

#for tups in testing_set:
#	y_pred.append(classifier.classify(tups[0]))
#	y_true.append(tups[1])


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

