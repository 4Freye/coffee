#import packages
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
from itertools import islice

#import stopwords
nltk.download('stopwords')

class FeatureCreate:
    def __init__(self, df, series):
        self.df = df
        if isinstance(series, str):
            self.series = self.df[series]
        else:
            return "Please input a string object (column name) for series"
            exit()
        #create list of words 
        self.corpus =  ' '.join(self.series).replace('.', ' ').replace(';', ' ').replace(',', ' ')

        # split() returns list of all the words in the string
        self.corpus_list = self.corpus.split()

        #words that aren't useful
        stop_words = set(stopwords.words('english'))
        stop_words.add('cup')

        #filter- this gets us only the interesting ones
        self.corpus_list = [w for w in self.corpus_list if not w.lower() in stop_words]


    def top_n_unigram(self, n = 20):
        # Python program to find the n most frequent words. then return dummy variables if frequent word is in series

        # Pass the list to instance of Counter class.
        top_n = Counter(self.corpus_list).most_common(n)
        #just get the words
        top_words = [tup[0] for tup in top_n]
        #convert to series
        top_words = pd.Series(top_words)
        #use apply to see if it's in description
        return top_words.apply(lambda x: self.series.str.contains(x, case=False)).set_index(top_words).transpose()
        
    def top_n_bigram(self, n = 10):
        # Python program to find the n most frequent bigrams. then return dummy variables if frequent bigram is in series

        # Pass the list to instance of Counter class.
        top_n =  Counter(zip(self.corpus_list, islice(self.corpus_list, 1, None))).most_common(n)

        #just get the words
        top_words = [tup[0] for tup in top_n]
        #convert to series
        top_words = pd.Series(top_words).apply(lambda x: ' '.join(x))
        #use apply to see if it's in description
        return top_words.apply(lambda x: self.series.str.contains(x, case=False)).set_index(top_words).transpose()