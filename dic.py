from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import pickle as pkl
import numpy as np

my_stop_words = text.ENGLISH_STOP_WORDS.union(['https', 'com','18cwlv', 'q1aqlp3sz5', 'breitbartreaderforandroid']).difference(['not', 'what', 'false', 'really'])
train = []
for (root, dir, files) in os.walk("dataset/train/"):
    for f in files:
        train += pkl.load(open('dataset/train/' + f, 'rb'))
vec = TfidfVectorizer(stop_words=my_stop_words, max_features=5000, max_df=0.8, min_df=3)
vec.fit(train)
# print(np.sum(vec.transform(train).toarray()[:5]))
f_train = open("dataset/train.pkl", 'wb')
f_test = open("dataset/test.pkl", 'wb')

for (root, dir, files) in os.walk("dataset/train/"):
    for f in files:
        l = pkl.load(open('dataset/train/' + f, 'rb'))
        pkl.dump(vec.transform(l).toarray(), f_train)
        # print(f.split('_')[1][0])
        pkl.dump(f.split('_')[1][0], f_train)

for (root, dir, files) in os.walk("dataset/test/"):
    for f in files:
        l = pkl.load(open('dataset/test/' + f, 'rb'))
        pkl.dump(vec.transform(l).toarray(), f_test)
        # print(f.split('_')[1][0])
        pkl.dump(f.split('_')[1][0], f_test)


