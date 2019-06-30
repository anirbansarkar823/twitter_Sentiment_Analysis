import io
import html.parser
import re

with io.open("result.txt",'r',encoding='utf8') as f:
		for tweet in f.readlines():
			# Remove extra spaces or newlines around the text
			tweet = tweet.strip()
			
			#removing html links
			regex = re.compile(r'http\S+')
			tweet = regex.sub('',tweet)
	
			regex = re.compile(r'www\S+')
			tweet = regex.sub('',tweet)
	
			regex = re.compile(r'https\S+')
			tweet = regex.sub('',tweet)
			
			#removing @tags
			regex = re.compile(r'@\S+')
			tweet = regex.sub('',tweet)
			
			#removing specific characters
			regex = re.compile(r'RT')
			tweet = regex.sub('',tweet)
			
			#splitng attached words
			tweet = " ".join(re.finadall('[A-Z][^A-Z]*',tweet))
			
			# Remove extra spaces or newlines around the text
			tweet = tweet.strip()
			
			# Reject tweets which are empty.
			if len(tweet) == 0:
				continue
				
			print(tweet)