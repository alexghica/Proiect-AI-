import nltk
import pandas as pd
import numpy as np
import sys
import copy
import random
from clasificator import KNN_classifier
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import TweetTokenizer
np.set_printoptions(threshold=sys.maxsize)   #sa mi afiseze toata matricea
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import fbeta_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
import string
import re
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import confusion_matrix

TRAIN_FILE = ''
TEST_FILE = ''
TXT_COL = 'text'
MSG_COL = 'misogynous'


def tokenize(text):

    #return nltk.WordPunctTokenizer().tokenize(text)

    '''tokeni1 = text.lower()
    tokeni2 = re.sub(r"http\S+", " ", tokeni1)
    tokeni3 = re.sub(r"[,?.!;&@#$]", " ", tokeni2)'''

    tokeni1 = re.sub(r"[,?.!;&@#$]", " ", text)
    tokeni2 = tokeni1.lower()
    tokeni3 = re.sub(r"http\S+", " ", tokeni2)
    #tokeni4 = re.sub(r"[0-9]", "", tokeni3) #test

    tokeni = nltk.TweetTokenizer(strip_handles=True, reduce_len=True).tokenize(tokeni3) #prezerve case e default True
    #tokeni = RegexpTokenizer('\w+|\$[\d\.]+|\S+').tokenize(tokeni4)
    #tokeni = RegexpTokenizer("[\w']+").tokenize(tokeni3)
    '''for i in tokeni:
        if i in string.punctuation:
            tokeni.remove(i)'''
    for i in tokeni:
        if len(i)>20 or len(i)<3:   # cu 3 e mai bine
            tokeni.remove(i)
    return tokeni


def get_representation(toate_cuvintele, how_many):
    '''Extract the first most common words from a vocabulary
    and return two dictionaries: word to index and index to word
    '''
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx,(cuvant, frecv) in enumerate(most_comm):
            wd2idx[cuvant] = idx
            idx2wd[idx] = cuvant
    return wd2idx, idx2wd


def get_corpus_vocabulary(corpus):
    '''Write a function to return all the words in a corpus.
    lower-cased
    '''
    counter = Counter()
    for text in corpus:
        tokens = tokenize(text.lower())
        counter.update(tokens)
    return counter


def text_to_bow(text, wd2idx):
    '''Convert a text to a bag of words representation.
    '''
# Reprezentarea vectoriala a unui text

    features = np.zeros(len(wd2idx))
    for cuvant in tokenize(text):
        if cuvant in wd2idx:
            idx = wd2idx[cuvant]
            features[idx] += 1
    return features


def corpus_to_bow(corpus, wd2idx):
    '''Convert a corpus to a bag of words representation.
    '''
    all_features = np.zeros((len(corpus),len(wd2idx)))
    for i, text in enumerate(corpus):
        bow  = text_to_bow(text,wd2idx)
        all_features[i,:] = bow
    return all_features


def write_prediction(out_file, predictions):
    '''A function to write the predictions in a file.
    '''


def split(data, labels,  procentaj_valid = 0.25):
    
    #75% train, 25% valid
    #mai intai facem shuffle la date
    
    indici = np.arrange(len(labels))
    random.shuffle(indici)
    dim_train = round((1-procentaj_valid) * len(labels))
    train = data[indici][:dim_train]
    y_train = labels[indici][:dim_train]
    valid = data[indici][dim_train:]
    y_valid = labels[indici][dim_train:]
    return train,valid,y_train,y_valid


def cross_validate(k, data, labels):
    '''Split into k chunks
    iteration 0:
        chunk 0 is for validation, chunk[1:] for train
    iteration 1:
        chunk 1 is for validation, chunk[0] + chunk[2:] is for train
    ...
    iteration k:
        chunk k is for validation, chunk[:k] is for train'''
    chunk_size = int(len(labels) / k)
    indici = np.arange(0,len(labels))
    random.shuffle(indici)

    for i in range(0, len(labels), chunk_size):
        valid_indici = indici[i:i +chunk_size]
        right_side = indici[i+chunk_size:]
        left_side = indici[0:i]
        train_indici = np.concatenate([left_side, right_side])
        train = data[train_indici]
        valid = data[valid_indici]
        y_train = labels[train_indici]
        y_valid = labels[valid_indici]
        yield train, valid, y_train, y_valid



def acc(y_true, y_pred):
    return np.mean((y_true == y_pred).astype(int)) * 100


def main():
    pass

if __name__ == '__main__':
    main()




train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


toate_cuvintele = get_corpus_vocabulary(train_df['text'])
wd2idx, idx2wd = get_representation(toate_cuvintele,1500)  #50-KNN #1400 #2000/1500  # gauss 2000-0.838/1700-0.842     ->>1600Multi<--
                                                           # ->>1600Multi1400<--    1600-Complement 1000-Bernoulli
data = corpus_to_bow(train_df['text'],wd2idx)              # Complement: 1700,1500   SVC = 400 - 600(2) , 700(de incercat)
labels = train_df['label']

test_data = corpus_to_bow(test_df['text'],wd2idx)

#print(toate_cuvintele.most_common(10))


date_noi1 = data[:4500, :]
date_test1 = data[4500:5000, :]
labels_noi1 = train_df['label'][:4500]
test_labels1 = train_df['label'][4500:5000]


'''clf1 = MultinomialNB()  # 0.814 acc si 0.82681 F1     cu Multinomial
clf1.fit(date_noi1,labels_noi1)
predictii1 = clf1.predict(date_test1)
print((predictii1==test_labels1).mean())
print("F1:",metrics.f1_score(test_labels1, clf1.predict(date_test1)))'''
clf1 = ComplementNB()
clf1.fit(date_noi1,labels_noi1)
predictii1 = clf1.predict(date_test1)
print((predictii1==test_labels1).mean())
print("F1:",metrics.f1_score(test_labels1, clf1.predict(date_test1)))


#clf2 = SVC(C=3)        #0.846 , F1=0.85871 , media 0.87028(cu 2) -  toate cu 800
clf2 = ComplementNB() #0.848 ,  F1 = 0.85977  , media 0.8687( cu 3)
#clf2 = MultinomialNB()  #0.844 ,  F1 = 0.8555   , media 0.8659 (cu 4)
                       #0.852 ,  F1 = 0.86346  , media 0.86619(cu 3 si 900)
                       #0.854 ,  F1 = 0.8650   , media 0.86607(cu 3 si 1000)


matrice_totala = np.zeros((2,2))
#for i in range(6):
scoruri = []
for train, valid, y_train, y_valid in cross_validate(10,data,labels):
    clf2.fit(train,y_train)
    predictii2 = clf2.predict(valid)
    scor = fbeta_score(y_valid, predictii2, beta =1)

    matrice_conf = np.zeros((2,2))
    for true_lbl, predi in zip(predictii2, y_valid):
        matrice_conf[true_lbl, predi] += 1
    matrice_totala += matrice_conf

    print(scor)
    scoruri.append(scor)

print("Media scorurilor este: ")
print(np.mean(scoruri), ' ', np.std(scoruri))
print(matrice_totala)



#Pentru submisie
clf = ComplementNB()
clf.fit(data,labels)
pred = clf.predict(test_data)
'''clf = MultinomialNB()
clf.fit(data,labels)
pred = clf.predict(test_data)'''


# Creare submisie
'''testId = np.arange(5001,6001)

print(testId[0:10])
print(pred[0:10])

print(testId.shape)
print(pred.shape)'''

#np.savetxt("E:\PYTHON\PYCHARM student\FISIERE PYCHARM\ProiectAI\submisie_Kaggle_30" ,
#           np.stack((testId,pred)).T, fmt ="%d", delimiter=',', header="id,label", comments='')

#Matrice confuzie
'''matrice_confuzie = np.zeros((2,2))
for true_lbl, predi in zip(predictii1,test_labels1):
    matrice_confuzie[true_lbl,predi] +=1
print(matrice_confuzie)
print(confusion_matrix(predictii1, test_labels1))'''