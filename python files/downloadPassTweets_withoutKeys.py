# The main module will download and pass the randomly generated tweets created to the already working model

import tweepy, codecs
import re
from classifyUsingPickle import classify_tweets
import os

# filling in my Twitter credentials
#put your own keys
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# let Tweepy set up an instance of the Rest API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# fill in the search query, and store the result in a variable
query_C = input("Enter your sentiment query\n")
limit_Q = int(input("enter the upper limit\n"))



#get the file from the query itself
dirName = "W:/works/FOCUS/FINAL_YEAR_PROJECT/Anirban/works/end_sem_final/textResults"
if not os.path.exists(dirName):
    os.mkdir(dirName)
else:    
    pass
    
    
fileName = re.sub(r'\W','',query_C,flags=re.I)
fileName = fileName + ".txt"
fileName = fileName.strip()
fileName = "W:/works/FOCUS/FINAL_YEAR_PROJECT/Anirban/works/end_sem_final/textResults" +'/'+fileName

#file = codecs.open(fileName,'w',encoding='utf8')

results = api.search(q = query_C, lang = "en", result_type = "recent", count = limit_Q)
#use the codecs library to write the text of the Tweets to a .txt file

file = codecs.open(fileName,"w")
for result in results:
	file.write(result.text)
	file.write("\n")
file.close()

#to search a number of queries
#last_id = None
#result = True
#while result:
#    results = api.search(q=query_C,lang="en", count=limit_Q, tweet_mode='extended', max_id=last_id)
#    for resul in results:
#        file = codecs.open(fileName,'a',encoding='utf8')
#        file.write(resul.text)
#        file.write("\n")
#    file.close()
    # we subtract one to not have the same again.
#    last_id = result[-1]._json['id'] - 1






#for classification
classify_tweets(fileName)