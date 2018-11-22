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
        self.POS_TAGS_TO_AUG = ['JJS', 'RBS']#['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']#['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'] 
        self.pos_corpus = []
        self.neg_corpus = []
        self.test_corpus = []
        self.train_vs_test_cutoff = 0.7
        self.train_vs_valid_cutoff = 0.7

        self.pos_aug_corpus = []
        self.neg_aug_corpus = []

        self.train_pos_tmp = []
        self.train_neg_tmp = []

        self.mis_classification = []
        print 'Movies review classifier ready for work!!!'

    def read_data(self):
        print 'READING DATA\n'
        # tokenizes each sentence/review, puts it in a separate list
        # and labels 'pos' or 'neg' after each list
        with open(self.POS_REVIEW_FILE_PATH, 'r') as sentences:
            self.pos_corpus = sentences.readlines()
        
        with open(self.NEG_REVIEW_FILE_PATH, 'r') as sentences:
            self.neg_corpus = sentences.readlines()

    def split_data(self):
        print 'SPLITTING DATA\n'
        # self.pos_corpus = self.pos_corpus + self.pos_aug_corpus
        # self.neg_corpus = self.neg_corpus + self.neg_aug_corpus

        full_corpus = self.pos_corpus + self.neg_corpus
        y_all = ['pos']*len(self.pos_corpus) + ['neg']*len(self.neg_corpus)

        pos_train_vs_test_cutoff = int(math.floor(len(self.pos_corpus)*self.train_vs_test_cutoff))
        neg_train_vs_test_cutoff = int(math.floor(len(self.neg_corpus)*self.train_vs_test_cutoff))
        
        X_train_pos, X_train_neg = self.pos_corpus[:pos_train_vs_test_cutoff], self.neg_corpus[:neg_train_vs_test_cutoff]
        
        self.train_pos_tmp = X_train_pos
        self.train_neg_tmp = X_train_neg

        self.augment_corpus()

        X_train_pos = X_train_pos + self.pos_aug_corpus
        X_train_neg = X_train_neg + self.neg_aug_corpus

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

    def get_vectors_tfidf_vectorizer(self, params, train_sample, test_sample):
        vectorizer = TfidfVectorizer(
                                encoding='latin-1',
                                token_pattern=r'[\w]+|[.,!?;]',
                                ngram_range=(
                                    params['ngram_min'] if params['ngram_min'] in params else 1, 
                                    params['ngram_max'] if params['ngram_max'] in params else 2),
                                stop_words=params['stop_words'] if params['stop_words'] in params else None, 
                                min_df=params['min_df'] if params['min_df'] in params else 1,
                                max_df = params['max_df'] if params['max_df'] in params else 1.0,
                                max_features = params['max_features'],
                                sublinear_tf=True,
                                use_idf=True)

        train_vetors = vectorizer.fit_transform(train_sample)
        test_vectors = vectorizer.transform(test_sample)
        return train_vetors, test_vectors, vectorizer

    def get_best_params(self):
        tune_lst = []
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':50000})
        return tune_lst

    def run_final_setup(self):
        params_lst = self.get_best_params()
        clr = MultinomialNB()
        clr_key = 'Naive Bayes'
        vectorizer = None

        split_train_vs_test_cutoffs = [0.7]
        split_train_vs_valid_cutoffs = [0.95]

        X_train, X_valid, y_train, y_valid, X_test, y_test = [], [], [], [], [], []
        max_vals = {'maxval': [0.2]}
        for params in params_lst:
            for i in split_train_vs_test_cutoffs:
                self.train_vs_test_cutoff = i
                for k in split_train_vs_valid_cutoffs:
                    self.train_vs_valid_cutoff = k
                    X_train, X_valid, y_train, y_valid, X_test, y_test = self.split_data()
                    print len(X_train), len(X_valid), len(X_test)
                    train_vecs, valid_vecs, vecter = self.get_vectors_tfidf_vectorizer(params, X_train, X_valid)
                    vectorizer = vecter
                    clr.fit(train_vecs, y_train)
                    y_predicted = clr.predict(valid_vecs)
                    acc = accuracy_score(y_valid, y_predicted)
                    if acc > max_vals['maxval'][0]:
                        max_vals['maxval'] = [acc, clr_key, i, k, params]
                    print clr_key
                    print 'Accuracy:', accuracy_score(y_valid, y_predicted)

        print ':::::performance on testing sample:::::'
        X_test_tr = vectorizer.transform(X_test)
        y_predicted_test = clr.predict(X_test_tr)
        print 'Accuracy:', accuracy_score(y_test, y_predicted_test)
        print confusion_matrix(y_test, y_predicted_test)
        # :::::preformance on testing sample:::::
        # Accuracy: 0.78875
        # [[1309  291]
        # [ 385 1215]]

        self.mis_classification = []
        for i in range(0, len(y_predicted_test)):
            if y_predicted_test[i] != y_test[i]:
                self.mis_classification.append([y_predicted_test[i], y_test[i], X_test[i]])

    def augment_sentence(self, sentence, corpus_polarity='pos'):
        # add 5-gram sentences around modified record
        # consider only sentences between certain length

        try:
            tagged_sen = nltk.pos_tag(word_tokenize(sentence))
        except UnicodeDecodeError:
            return
        aug_pos_sent = []
        aug_neg_sent = []
        neg_flag = False
        pos_flag = False 

        for i in range(len(tagged_sen)):
            syno_lst = [] 
            anto_lst = [] 
            if tagged_sen[i][1] in self.POS_TAGS_TO_AUG:               
                # tagged_sen[i][0], tagged_sen[i][1], 
                try:
                    for syn in wn.synsets(tagged_sen[i][0]): 
                        for l in syn.lemmas(): 
                            syno_lst.append(l.name().encode('utf-8'))
                         
                            if l.antonyms(): 
                                anto_lst.append(l.antonyms()[0].name().encode('utf-8'))

                except UnicodeDecodeError:
                    return
                if syno_lst != []:
                    if corpus_polarity == 'pos':
                        aug_pos_sent.append(syno_lst[0].encode('utf-8'))
                    else:
                        aug_neg_sent.append(syno_lst[0].encode('utf-8'))
                    pos_flag = True
                if anto_lst != []:
                    if corpus_polarity == 'pos':
                        aug_neg_sent.append(anto_lst[0].encode('utf-8'))
                    else:
                        aug_pos_sent.append(anto_lst[0].encode('utf-8'))
                    neg_flag = True

            else:
                aug_pos_sent.append(tagged_sen[i][0])
                aug_neg_sent.append(tagged_sen[i][0])
        # print ':::syno::: ', syno_lst, '\n'
        # print sentence, len(sentence.split()), '\n'
        # print aug_pos_sent, " ".join(aug_pos_sent), len(aug_pos_sent), '\n'
        # print aug_neg_sent, " ".join(aug_neg_sent), len(aug_neg_sent), '\n'
        # print ':::anto::: ', anto_lst, '\n'
        if pos_flag and (len(aug_pos_sent) >= len(sentence.split())):
            self.pos_aug_corpus.append(" ".join(aug_pos_sent))
        if neg_flag and (len(aug_neg_sent) >= len(sentence.split())):
            self.neg_aug_corpus.append(" ".join(aug_neg_sent))

    def augment_corpus(self):
        print 'AUGMENTING DATA\n'
        for s in self.train_pos_tmp:
            self.augment_sentence(s, 'pos')

        for s in self.train_neg_tmp:
            self.augment_sentence(s, 'neg')

    def main(self):
        # phase 1: read data from files;
        self.read_data()

        # phase 2: augment data
        # self.augment_corpus()

        # phase 3: run final train, with the optimal params and samples split ratios, as obtained in phase 3 & 4
        self.run_final_setup()


if __name__ == "__main__":
    mrc = MoviesReviewClassifier()
    print 'start classifying:'
    mrc.main()