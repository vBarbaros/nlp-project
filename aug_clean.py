import math, os
import nltk

from nltk import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from nltk.corpus import wordnet as wn


class MoviesReviewClassifier:

    def __init__(self):
        self.POS_REVIEW_FILE_PATH = os.path.join('rt-polaritydata', 'rt-polarity.pos')
        self.NEG_REVIEW_FILE_PATH = os.path.join('rt-polaritydata', 'rt-polarity.neg')
        self.POS_TAGS_TO_AUG = ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'] 
        self.pos_corpus = []
        self.neg_corpus = []
        self.test_corpus = []
        self.train_vs_test_cutoff = 0.7
        self.train_vs_valid_cutoff = 0.7
        self.mis_classification = []
        print 'Movies review classifier ready for work!!!'

    def read_data(self):
        # tokenizes each sentence/review, puts it in a separate list
        # and labels 'pos' or 'neg' after each list
        with open(self.POS_REVIEW_FILE_PATH, 'r') as sentences:
            self.pos_corpus = sentences.readlines()
        
        with open(self.NEG_REVIEW_FILE_PATH, 'r') as sentences:
            self.neg_corpus = sentences.readlines()

    def split_data(self):
        full_corpus = self.pos_corpus + self.neg_corpus
        y_all = ['pos']*len(self.pos_corpus) + ['neg']*len(self.neg_corpus)

        pos_train_vs_test_cutoff = int(math.floor(len(self.pos_corpus)*self.train_vs_test_cutoff))
        neg_train_vs_test_cutoff = int(math.floor(len(self.neg_corpus)*self.train_vs_test_cutoff))
        
        # X_train_tmp = self.pos_corpus[:pos_sample_cutoff] + self.neg_corpus[:neg_sample_cutoff]
        # y_train = ['pos']*len(self.pos_corpus[:pos_sample_cutoff]) + ['neg']*len(self.neg_corpus[:neg_sample_cutoff])

        X_train_pos, X_train_neg = self.pos_corpus[:pos_train_vs_test_cutoff], self.neg_corpus[:neg_train_vs_test_cutoff]
        y_train_pos, y_train_neg = ['pos']*len(self.pos_corpus[:pos_train_vs_test_cutoff]), ['neg']*len(self.neg_corpus[:neg_train_vs_test_cutoff])

        pos_train_vs_valid_cutoff = int(math.floor((len(X_train_pos))*self.train_vs_valid_cutoff))
        neg_train_vs_valid_cutoff = int(math.floor((len(X_train_neg))*self.train_vs_valid_cutoff))

        X_train = X_train_pos[:pos_train_vs_valid_cutoff] + X_train_neg[:neg_train_vs_valid_cutoff]
        y_train = ['pos']*len(X_train_pos[:pos_train_vs_valid_cutoff]) + ['neg']*len(X_train_neg[:neg_train_vs_valid_cutoff])
        X_valid = X_train_pos[pos_train_vs_valid_cutoff:] + X_train_neg[neg_train_vs_valid_cutoff:]
        y_valid = ['pos']*len(X_train_pos[pos_train_vs_valid_cutoff:]) + ['neg']*len(X_train_neg[neg_train_vs_valid_cutoff:])

        X_test = self.pos_corpus[pos_train_vs_test_cutoff:] + self.neg_corpus[neg_train_vs_test_cutoff:]
        y_test = ['pos']*len(self.pos_corpus[pos_train_vs_test_cutoff:]) + ['neg']*len(self.neg_corpus[neg_train_vs_test_cutoff:])
        self.test_corpus = X_test
        return X_train, X_valid, y_train, y_valid, X_test, y_test

    def augment_sentence(self, sentence):
        synonyms = [] 
        antonyms = [] 
        tagged_sen = nltk.pos_tag(word_tokenize(sentence))
        for item in tagged_sen:
            if item[1] in self.POS_TAGS_TO_AUG:               
                # print item[0], item[1], 
                for syn in wn.synsets(item[0]): 
                    for l in syn.lemmas(): 
                        synonyms.append(l.name()) 
                        if l.antonyms(): 
                            antonyms.append(l.antonyms()[0].name())

        print ':::syno::: ', synonyms, '\n'
        print ':::ante::: ', antonyms, '\n'

    def main(self):
        # phase 1: read data from files;
        self.read_data()

        # for i in self.pos_corpus:
        #     try:
        #         print i, nltk.pos_tag(word_tokenize(i.encode('utf-8')))
        #     except UnicodeDecodeError:
        #         pass
        
        # Tags to look at:
        # JJ(adjective), JJR(adj comparative), JJS(adj superlative)
        # RB(adverb), RBR(adverb comparative), RBS(adverb superlative), WRB(Wh-adverb)-???,

        
        # print self.pos_corpus[0], nltk.pos_tag(word_tokenize(self.pos_corpus[0]))
        # print self.pos_corpus[1], nltk.pos_tag(word_tokenize(self.pos_corpus[1]))
        # print self.pos_corpus[2], nltk.pos_tag(word_tokenize(self.pos_corpus[2]))
        self.augment_sentence(self.pos_corpus[0])
        self.augment_sentence(self.pos_corpus[1])
        self.augment_sentence(self.pos_corpus[2])

        # print nltk.tag.pos_tag(word_tokenize('I am doing great.'))


if __name__ == "__main__":
    mrc = MoviesReviewClassifier()
    print 'start classifying:'
    mrc.main()