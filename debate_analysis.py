# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:12:03 2016
Project: Topic Modelling for Presidential Debate
Author: Yongjun Zhang
Affiliation: School of Sociology, Univerisity of Arizona, Tucson, USA
Email: yongjunzhang@email.arizona.edu
Address: Social Science Building 412, Tucson, USA, 85719
Wechat：joshzyj
"""
import os 
import re,string
import csv
import glob
from datetime import datetime
import pandas as pd
import csv
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

def get_text(file):
    """ Extract txt files, return cleaned hilary and trump debate transcript"""
    doc=open(file, encoding= "utf-8").read()
    doc=re.sub(r'\n', ' ', doc)
    doc=re.sub(r'WALLACE:', 'joshzyj WALLACE:',doc)
    doc=re.sub(r'CLINTON:', 'joshzyj CLINTON:',doc)
    doc=re.sub(r'TRUMP:', 'joshzyj TRUMP:',doc)
    doc=re.sub(r'$', 'joshzyj',doc)
    hillary_doc=re.findall(r'CLINTON: (.*?)joshzyj',str(doc))
    trump_doc=re.findall(r'TRUMP: (.*?)joshzyj',str(doc))
    return hillary_doc, trump_doc

def text_to_words(raw_text):
    """
    Function to convert a raw text to a string of words
    The input is a single string (a raw text), and 
    the output is a single list containing all words with different sentences
    """
    try:
        # 1. Remove non-letters        
        letters_only = re.sub("[^a-zA-Z_]", " ", raw_text) 
        # 2. Convert to lower case, split into individual words
        words = letters_only.lower().split()                             
        # 3. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("english"))                  
        # 4. Remove stop words
        meaningful_words = [w for w in words if not w in stops]   
        # 5. Stemming words
        lmtzr = WordNetLemmatizer()
        lemmatized_words = [lmtzr.lemmatize(w) for w in meaningful_words]
        # 6. Join the words back into one string separated by space, 
        # and return the result.
        return   " ".join(lemmatized_words)
    except:
        pass

def get_words(file):
    file=pd.DataFrame(file)
    num_names = file[0].size
    words=[]
    for i in range(0, num_names):
        if( (i+1)%2 == 0 ):
            print('Review %d of %d\n' % ( i + 1, num_names))
        words.append(text_to_words(file[0][i]))
    return words


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()



os.chdir(r"/Users/joshzyj/GoogleDrive/2016 Fall/2016Winter/soc315/")
file="transcript_all_debate3.txt"
hillary_doc, trump_doc=get_text(file)
hillary_words=get_words(hillary_doc)
trump_words=get_words(trump_doc)


count_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = ['crosstalk'], max_features = 500)

tfidf_vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = ['crosstalk'], max_features = 500, ngram_range=(1,2))


tfidf = tfidf_vectorizer.fit_transform(trump_words)

n_topics = 5
n_top_words = 10

lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tfidf)

tf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)        
        
nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)

print_top_words(nmf, tf_feature_names, n_top_words)

##Further analysis

