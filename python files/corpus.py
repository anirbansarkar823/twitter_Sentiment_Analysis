import tweepy, codecs

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
results = api.search(q = "Text Mining", lang = "en", result_type = "recent", count = 100)

#use the codecs library to write the text of the Tweets to a .txt file

file = codecs.open("result_Text_mine.txt","w","utf-8")
for result in results:
	file.write(result.text)
	file.write("\n")
file.close()