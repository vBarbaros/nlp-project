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

from itertools import compress, product

import gensim  


class MoviesReviewClassifier:

    def __init__(self, use_word2vec):
        self.POS_REVIEW_FILE_PATH = os.path.join('rt-polaritydata', 'rt-polarity.pos')
        self.NEG_REVIEW_FILE_PATH = os.path.join('rt-polaritydata', 'rt-polarity.neg')
        self.pos_corpus = []
        self.neg_corpus = []
        self.test_corpus = []
        self.train_vs_test_cutoff = 0.7
        self.train_vs_valid_cutoff = 0.7
        self.use_word2vec = use_word2vec

        self.pos_aug_corpus = []
        self.neg_aug_corpus = []

        self.train_pos_tmp = []
        self.train_neg_tmp = []

        self.mis_classification = []
        if use_word2vec:
            self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
        print 'Movies review classifier ready for work!!!'

    def read_data(self):
        print 'READING DATA\n'
        # tokenizes each sentence/review, puts it in a separate list
        # and labels 'pos' or 'neg' after each list
        with open(self.POS_REVIEW_FILE_PATH, 'r') as sentences:
            self.pos_corpus = sentences.readlines()
        
        print self.pos_corpus[:3]

        with open(self.NEG_REVIEW_FILE_PATH, 'r') as sentences:
            self.neg_corpus = sentences.readlines()

    def read_data_imdb(self):
        print 'READING DATA\n'
        # tokenizes each sentence/review, puts it in a separate list
        # and labels 'pos' or 'neg' after each list

        pos_train_rev_path = 'aclimdb/train/pos/'
        files_pos_train_review = os.listdir(pos_train_rev_path)#['0_9.txt', '1_7.txt', '2_9.txt']
        for fl in files_pos_train_review:
            with open(pos_train_rev_path + fl, 'r') as sentences:
                # read from file and put the entire content of it as single string
                self.pos_corpus.append(sentences.readlines()[0])
        
        # pos_test_rev_path = 'aclimdb/test/pos/'
        # files_pos_test_review = os.listdir(pos_test_rev_path)
        # for fl in files_pos_test_review:
        #     with open(pos_test_rev_path + fl, 'r') as sentences:
        #         self.pos_corpus.append(sentences.readlines()[0])

        neg_train_rev_path = 'aclimdb/train/neg/'
        files_neg_train_review = os.listdir(neg_train_rev_path)
        for fl in files_neg_train_review:
            with open(neg_train_rev_path + fl, 'r') as sentences:
                # read from file and put the entire content of it as single string
                self.neg_corpus.append(sentences.readlines()[0])
        
        # neg_test_rev_path = 'aclimdb/test/neg/'
        # files_neg_test_review = os.listdir(neg_test_rev_path)
        # for fl in files_neg_test_review:
        #     with open(neg_test_rev_path + fl, 'r') as sentences:
        #         self.neg_corpus.append(sentences.readlines()[0])

    def split_data(self, pos_combination):
        print 'SPLITTING DATA\n'

        full_corpus = self.pos_corpus + self.neg_corpus
        y_all = ['pos']*len(self.pos_corpus) + ['neg']*len(self.neg_corpus)

        pos_train_vs_test_cutoff = int(math.floor(len(self.pos_corpus)*self.train_vs_test_cutoff))
        neg_train_vs_test_cutoff = int(math.floor(len(self.neg_corpus)*self.train_vs_test_cutoff))
        
        X_train_pos, X_train_neg = self.pos_corpus[:pos_train_vs_test_cutoff], self.neg_corpus[:neg_train_vs_test_cutoff]
        
        y_train_pos, y_train_neg = ['pos']*len(self.pos_corpus[:pos_train_vs_test_cutoff]), ['neg']*len(self.neg_corpus[:neg_train_vs_test_cutoff])

        pos_train_vs_valid_cutoff = int(math.floor((len(X_train_pos))*self.train_vs_valid_cutoff))
        neg_train_vs_valid_cutoff = int(math.floor((len(X_train_neg))*self.train_vs_valid_cutoff))

        self.train_pos_tmp = X_train_pos[:pos_train_vs_valid_cutoff]
        self.train_neg_tmp = X_train_neg[:neg_train_vs_valid_cutoff]

        if len(pos_combination) != 0:
            self.pos_aug_corpus, self.neg_aug_corpus = [], []
            self.augment_corpus(pos_combination) # augment based only on the training set 

        X_train_pos_aug = X_train_pos[:pos_train_vs_valid_cutoff] + self.pos_aug_corpus
        X_train_neg_aug = X_train_neg[:neg_train_vs_valid_cutoff] + self.neg_aug_corpus

        X_train = X_train_pos_aug + X_train_neg_aug
        y_train = ['pos']*len(X_train_pos_aug) + ['neg']*len(X_train_neg_aug)


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
    
    
    def combinations(self, items):
        return ( frozenset(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )

    def run_final_setup(self):
        params_lst = self.get_best_params()
        clr = MultinomialNB()
        clr_key = 'Naive Bayes'
        vectorizer = None

        split_train_vs_test_cutoffs = [0.5]
        split_train_vs_valid_cutoffs = [0.7]

        # generate all combinations of pos considered
        pos_to_consider = ['JJ', 'JJS', 'RBS'] #['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'] #
        pos_combinations = self.combinations(pos_to_consider)


        X_train, X_valid, y_train, y_valid, X_test, y_test = [], [], [], [], [], []
        max_vals = {'maxval': [0.2]}
        results_combinations = {}
        for params in params_lst:
            for i in split_train_vs_test_cutoffs:
                self.train_vs_test_cutoff = i
                for k in split_train_vs_valid_cutoffs:
                    for pos_combination in pos_combinations:
                        self.train_vs_valid_cutoff = k
                        X_train, X_valid, y_train, y_valid, X_test, y_test = self.split_data(pos_combination)
                        print len(X_train), len(X_valid), len(X_test)
                        train_vecs, valid_vecs, vecter = self.get_vectors_tfidf_vectorizer(params, X_train, X_valid)
                        vectorizer = vecter
                        clr.fit(train_vecs, y_train)
                        y_predicted = clr.predict(valid_vecs)
                        acc = accuracy_score(y_valid, y_predicted)
                        if acc > max_vals['maxval'][0]:
                            max_vals['maxval'] = [acc, clr_key, i, k, params, pos_combination]

                        results_combinations[(i, k, pos_combination)] = acc
                        print pos_combination
                        print clr_key
                        print 'Accuracy:', accuracy_score(y_valid, y_predicted)

        print ':::::performance on testing sample:::::'
        X_test_tr = vectorizer.transform(X_test)
        y_predicted_test = clr.predict(X_test_tr)
        print 'Accuracy:', accuracy_score(y_test, y_predicted_test)
        print confusion_matrix(y_test, y_predicted_test)
        print max_vals['maxval']
        # :::::preformance on testing sample:::::
        # Accuracy: 0.78875
        # [[1309  291]
        # [ 385 1215]]

        self.mis_classification = []
        for i in range(0, len(y_predicted_test)):
            if y_predicted_test[i] != y_test[i]:
                self.mis_classification.append([y_predicted_test[i], y_test[i], X_test[i]])

    def word2vec_implementation(self, sentence, word_index):
        # generate the top 5 positively related words
        similar_words = self.word2vec_model.most_similar(positive=sentence[word_index])
        list_generated_sentences = []
        for similar_word in similar_words:
            new_sentence = sentence
            new_sentence[word_index] = similar_word
            list_generated_sentences.append(new_sentence)
        return list_generated_sentences

    def augment_sentence(self, sentence, pos_combination, corpus_polarity='pos', sent_len_threshold=0, use_n_grams_sents=0, nth_word=0):
        # add 5-gram sentences around modified record
        # consider only sentences between certain length

        try:
            tokenized_sentence = word_tokenize(sentence)
            tagged_sen = nltk.pos_tag(tokenized_sentence)
        except UnicodeDecodeError:
            return
        aug_pos_sent = []
        aug_pos_sent_at_idx = []
        aug_neg_sent = []
        aug_neg_sent_at_idx = []
        neg_flag = False
        pos_flag = False 

        '''
        augmented_sentences_w2v = []
        for i in range(len(tagged_sen)):
            if tagged_sen[i][1] in self.POS_TAGS_TO_AUG:
                augmented_sentences_w2v.append(self.word2vec_implementation(tokenized_sentence, i))
        
        return augmented_sentences_w2v
        '''

        for i in range(len(tagged_sen)):
            syno_lst = [] 
            anto_lst = [] 
            if tagged_sen[i][1] in pos_combination:             
                # tagged_sen[i][0], tagged_sen[i][1], 

                if not self.use_word2vec:
                    try:
                        for syn in wn.synsets(tagged_sen[i][0]): 
                            for l in syn.lemmas(): 
                                syno_lst.append(l.name().encode('utf-8'))
                            
                                if l.antonyms(): 
                                    anto_lst.append(l.antonyms()[0].name().encode('utf-8'))

                    except UnicodeDecodeError:
                        return
                else:
                    # take whichever of the negative or positive score is the highest
                    pos_found, neg_found = False, False
                    try:
                        pos_similar_w2v = self.word2vec_model.most_similar(positive=tagged_sen[i][0], topn=1)[0]
                        pos_found = True
                    except :
                        pass
                    try:
                        neg_similar_w2v = self.word2vec_model.most_similar(negative=tagged_sen[i][0], topn=1)[0]
                        neg_found = True
                    except:
                        pass
                    
                    if pos_found and not neg_found:
                        syno_lst.append(pos_similar_w2v[0].encode("utf-8"))
                    elif not pos_found and neg_found:
                        anto_lst.append(neg_similar_w2v[0].encode("utf-8"))
                    elif not pos_found and not neg_found:
                        pass
                    else:
                        if pos_similar_w2v[1] > neg_similar_w2v[1]:
                            syno_lst.append(pos_similar_w2v[0].encode("utf-8"))
                        else:
                            anto_lst.append(neg_similar_w2v[0].encode("utf-8"))
                        



                if syno_lst != []:
                    if corpus_polarity == 'pos':
                        if nth_word < len(syno_lst):
                            aug_pos_sent.append(syno_lst[nth_word].encode('utf-8'))
                        else:
                            aug_pos_sent.append(syno_lst[0].encode('utf-8'))
                        aug_pos_sent_at_idx.append(i)
                    else:
                        if nth_word < len(syno_lst):
                            aug_neg_sent.append(syno_lst[nth_word].encode('utf-8'))
                        else:
                            aug_neg_sent.append(syno_lst[0].encode('utf-8'))
                        aug_neg_sent_at_idx.append(i)
                    pos_flag = True
                if anto_lst != []:
                    if corpus_polarity == 'pos':
                        if nth_word < len(anto_lst):
                            aug_neg_sent.append(anto_lst[nth_word].encode('utf-8'))
                        else:
                            aug_neg_sent.append(anto_lst[0].encode('utf-8'))
                        aug_neg_sent_at_idx.append(i)
                    else:
                        if nth_word < len(anto_lst):
                            aug_pos_sent.append(anto_lst[nth_word].encode('utf-8'))
                        else:
                            aug_pos_sent.append(anto_lst[0].encode('utf-8'))
                        aug_pos_sent_at_idx.append(i)
                    neg_flag = True

                    
            else:
                aug_pos_sent.append(tagged_sen[i][0])
                aug_neg_sent.append(tagged_sen[i][0])


        if sent_len_threshold != 0:
            # if threshold given => take only those sents up to the value
            if pos_flag and (len(aug_pos_sent) <= sent_len_threshold):
                self.pos_aug_corpus.append(" ".join(aug_pos_sent))
            if neg_flag and (len(aug_neg_sent) <= sent_len_threshold):
                self.neg_aug_corpus.append(" ".join(aug_neg_sent))
        elif use_n_grams_sents != 0:
            for idx in aug_pos_sent_at_idx:
                self.pos_aug_corpus.append(" ".join(aug_pos_sent[idx - use_n_grams_sents : idx + use_n_grams_sents]))

            for idx in aug_neg_sent_at_idx:
                self.neg_aug_corpus.append(" ".join(aug_neg_sent[idx - use_n_grams_sents : idx + use_n_grams_sents]))
        else:
            # just take the senteces that have been augmented
            if pos_flag and (len(aug_pos_sent) >= len(sentence.split())):
                self.pos_aug_corpus.append(" ".join(aug_pos_sent))
            if neg_flag and (len(aug_neg_sent) >= len(sentence.split())):
                self.neg_aug_corpus.append(" ".join(aug_neg_sent))

    def augment_corpus(self, pos_combination):
        print 'AUGMENTING DATA\n'
        for s in self.train_pos_tmp:
            self.augment_sentence(s, pos_combination,'pos', sent_len_threshold=0, use_n_grams_sents=4, nth_word=3)

        for s in self.train_neg_tmp:
            self.augment_sentence(s, pos_combination,'neg', sent_len_threshold=0, use_n_grams_sents=4, nth_word=3)

    def main(self):
        # phase 1: read data from files;
        # self.read_data()

        self.read_data_imdb()

        # phase 3: run final train, with the optimal params and samples split ratios, as obtained in phase 3 & 4
        self.run_final_setup()


if __name__ == "__main__":
    mrc = MoviesReviewClassifier(use_word2vec=False)
    print 'start classifying:'
    mrc.main()