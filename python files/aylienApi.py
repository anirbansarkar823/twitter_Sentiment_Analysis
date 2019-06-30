from aylienapiclient import textapi
import csv, io

#put your own keys
client = textapi.Client('','')
#sentiment = client.Sentiment({'text':'The condition of Arunachal is very dangerous. '})
#print(sentiment)

with io.open('Trump_Tweets.csv','w',encoding='utf8', newline='') as csvfile:
	csv_writer = csv.writer(csvfile)
	csv_writer.writerow(["Tweet","Sentiment"])
	
	with io.open("result_Text_mine.txt",'r',encoding='utf8') as f:
		for tweet in f.readlines():
			# Remove extra spaces or newlines around the text
			tweet = tweet.strip()
			
			# Reject tweets which are empty.
			if len(tweet) == 0:
				print('skipped')
				continue
				
			print(tweet)
			
			#Make call to ANYLIEN Text API
			sentiment = client.Sentiment({'text': tweet})
			
			#write the sentiment result into csv file
			csv_writer.writerow([sentiment['text'],sentiment['polarity']])
			