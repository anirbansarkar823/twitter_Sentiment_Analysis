# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:41:13 2019

@author: Enter My World
"""

import re, itertools
tweet = """ThisIsAwesome in th ebeach Here. This is insan in th insan soooooo happppyyyyy
is my new line getting maintained
"""

#split words
tweet = " ".join(re.findall('[A-Z][^A-Z]*',tweet))
print(tweet)

#standardising words
tweet =  ''.join(''.join(s)[:2] for _,s in itertools.groupby(tweet))
print(tweet)
#to test the working of spelling checker
from spellchecker import SpellChecker
spell = SpellChecker()
#find misspelled words
from nltk.tokenize import word_tokenize
tweet_tok = word_tokenize(tweet)
print(tweet_tok)
misspelled = spell.unknown(tweet_tok)
for word in misspelled:
    print(word)
    tweet = re.sub(word, spell.correction(word),tweet)
    
print(tweet)