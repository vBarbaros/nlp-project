import math, os
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Author: Victor Barbaros


class MoviesReviewClassifier:

    def __init__(self):
        self.POS_REVIEW_FILE_PATH = os.path.join('rt-polaritydata', 'rt-polarity.pos')
        self.NEG_REVIEW_FILE_PATH = os.path.join('rt-polaritydata', 'rt-polarity.neg')
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

    def split_data_count_vectorizer(self, vectorizer):
        full_corpus = self.pos_corpus + self.neg_corpus
        y_all = ['pos']*len(self.pos_corpus) + ['neg']*len(self.neg_corpus)

        pos_train_vs_test_cutoff = int(math.floor(len(self.pos_corpus)*self.train_vs_test_cutoff))
        neg_train_vs_test_cutoff = int(math.floor(len(self.neg_corpus)*self.train_vs_test_cutoff))
        
        X_train_tmp = self.pos_corpus[:pos_train_vs_test_cutoff] + self.neg_corpus[:neg_train_vs_test_cutoff]
        y_train = ['pos']*len(self.pos_corpus[:pos_train_vs_test_cutoff]) + ['neg']*len(self.neg_corpus[:neg_train_vs_test_cutoff])

        X_test = self.pos_corpus[pos_train_vs_test_cutoff:] + self.neg_corpus[neg_train_vs_test_cutoff:]
        self.test_corpus = X_test
        y_test = ['pos']*len(self.pos_corpus[pos_train_vs_test_cutoff:]) + ['neg']*len(self.neg_corpus[neg_train_vs_test_cutoff:])
        
        X_all = vectorizer.fit_transform(X_train_tmp).toarray()
        y_all = y_train
        X_train, X_valid, y_train, y_valid = train_test_split(
                                                            X_all, y_all, 
                                                            test_size=1.0-self.train_vs_valid_cutoff,
                                                            random_state=101)
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

    def get_vectors_count_vectorizer(self, params):
        # re0 = r'[\w]+|[.,!?;]' #Accuracy: 0.7792877225866917
        # re1 = r'[^\d\W]+|[.,!?;]/g' #Accuracy: 0.7755388940955952
        # re2 = r'[A-Za-z0-9]+|[.,!?;]' #Accuracy: 0.7792877225866917
        vectorizer = CountVectorizer(
                                encoding='latin-1',
                                analyzer='word',
                                token_pattern=r'[\w]+|[.,!?;]',
                                ngram_range=(
                                    params['ngram_min'] if params['ngram_min'] in params else 1,
                                    params['ngram_max'] if params['ngram_max'] in params else 2),
                                stop_words=params['stop_words'] if params['stop_words'] in params else None,
                                min_df=params['min_df'] if params['min_df'] in params else 1,
                                max_df = params['max_df'] if params['max_df'] in params else 1.0,
                                max_features = params['max_features'], 
                                lowercase=False)
        return vectorizer

    def get_classifiers(self):
        lgr_clr = LogisticRegression() 
        mnb_clr = MultinomialNB()
        svm_clr = LinearSVC()
        return {'log_reg': lgr_clr, 'naive_bayes': mnb_clr, 'svm': svm_clr}

    def get_params_model_selection(self):
        model_select_params = []
        model_select_params.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None,'min_df': 3,'max_df':0.95,'max_features':None})
        model_select_params.append({'ngram_min': 1,'ngram_max': 2,'stop_words': None,'min_df': 3,'max_df':0.95,'max_features':None})
        model_select_params.append({'ngram_min': 2,'ngram_max': 2,'stop_words': None,'min_df': 3,'max_df':0.95,'max_features':None})
        model_select_params.append({'ngram_min': 1,'ngram_max': 1,'stop_words': 'english','min_df': 3,'max_df':0.95,'max_features':None})
        model_select_params.append({'ngram_min': 1,'ngram_max': 2,'stop_words': 'english','min_df': 3,'max_df':0.95,'max_features':None})
        model_select_params.append({'ngram_min': 2,'ngram_max': 2,'stop_words': 'english','min_df': 3,'max_df':0.95,'max_features':None})
        return model_select_params

    def get_params_tune_model(self):
        tune_lst = []
        # keep params from previous phase and explore 'max_features'
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':1})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':2})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':3})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':4})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':5})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':10})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':20})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':30})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':40})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':50})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':100})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':200})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':400})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':800})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':1000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':2000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':4000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':5000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':10000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':15000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':20000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':30000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':40000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':50000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':100000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':150000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':200000})
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':300000})
        return tune_lst

    def get_best_params(self):
        tune_lst = []
        tune_lst.append({'ngram_min': 1,'ngram_max': 1,'stop_words': None, 'min_df': 3, 'max_df': 0.95, 'max_features':50000})
        return tune_lst

    def fine_tune_params_on_selected_model_and_vectorizer(self):
        X_train, X_valid, y_train, y_valid, X_test, y_test = self.split_data()
        params_lst = self.get_params_tune_model()
        clr = MultinomialNB()
        clr_key = 'Naive Bayes'
        vectorizer = None

        max_vals = {'maxval': [0.2]}
        for params in params_lst:
            print params
            train_vecs, valid_vecs, vecter = self.get_vectors_tfidf_vectorizer(params, X_train, X_valid)
            vectorizer = vecter
            clr.fit(train_vecs, y_train)
            y_predicted = clr.predict(valid_vecs)
            acc = accuracy_score(y_valid, y_predicted)
            if acc > max_vals['maxval'][0]:
                max_vals['maxval'] = [acc, clr_key, params]
            print clr_key
            print 'Accuracy:', accuracy_score(y_valid, y_predicted)

        print '::::::::::done::::::::::'
        print 'max acc: ', max_vals['maxval']
        # ::::::::::done::::::::::
        # max acc:  [0.7674107142857143, 'Naive Bayes', {'stop_words': None, 'ngram_min': 1, 'ngram_max': 1, 
        #'max_df': 0.95, 'min_df': 3, 'max_features': 50000}]

    def tune_classfier_samples_size(self):
        params_lst = self.get_best_params()
        clr = MultinomialNB()
        clr_key = 'Naive Bayes'
        vectorizer = None

        split_train_vs_test_cutoffs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        split_train_vs_valid_cutoffs = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

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

        print '::::::::::done::::::::::'
        print 'max acc: ', max_vals['maxval']
        # ::::::::::done:::::::::: best results on Validation
        # max acc:  [0.8101604278074866, 'Naive Bayes', 0.7, 0.95, {'stop_words': None, 'ngram_min': 1, 
        # 'ngram_max': 1, 'max_df': 0.95, 'min_df': 3, 'max_features': 50000}]

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

        print ':::::preformance on testing sample:::::'
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

    def check_content_misclassif(self):
        for i in range(0, len(self.mis_classification)):
            print self.mis_classification[i]

    def run_trials_select_model(self):
        X_train, X_valid, y_train, y_valid, X_test, y_test = self.split_data()
        params_lst = self.get_params_model_selection()
        clfrs_dict = self.get_classifiers()
        vectorizer = None

        max_vals = {'maxval': [0.2]}
        for params in params_lst:
            print params
            for clr_key, clr in clfrs_dict.iteritems():
                train_vecs, valid_vecs, vecter = self.get_vectors_tfidf_vectorizer(params, X_train, X_valid)
                vectorizer = vecter
                clr.fit(train_vecs, y_train)
                y_predicted = clr.predict(valid_vecs)
                acc = accuracy_score(y_valid, y_predicted)
                if acc > max_vals['maxval'][0]:
                    max_vals['maxval'] = [acc, clr_key, params]
                print clr_key
                print 'Accuracy:', accuracy_score(y_valid, y_predicted)

        print 'done::::::tfidf_vectorizer'
        print max_vals['maxval']
        # done::::::tfidf_vectorizer
        # [0.7660714285714286, 'naive_bayes', {'stop_words': None, 'ngram_min': 1, 'ngram_max': 1, 
        # 'max_df': 0.95, 'min_df': 3, 'max_features': None}]
        
        max_vals = {'maxval': [0.2]}
        for params in params_lst:
            print params
            for clr_key, clr in clfrs_dict.iteritems():
                vectorizer = self.get_vectors_count_vectorizer(params)
                X_train, X_valid, y_train, y_valid, X_test, y_test = self.split_data_count_vectorizer(vectorizer)
                clr.fit(X=X_train, y=y_train)
                y_predicted = clr.predict(X_valid)
                acc = accuracy_score(y_valid, y_predicted)
                if acc > max_vals['maxval'][0]:
                    max_vals['maxval'] = [acc, clr_key, params]
                print clr_key
                print 'Accuracy:', accuracy_score(y_valid, y_predicted)

        print 'done::::::count_vectorizer'
        print max_vals['maxval']
        # done::::::count_vectorizer
        # [0.7597141581062975, 'naive_bayes', {'stop_words': None, 'ngram_min': 1, 'ngram_max': 1, 
        # 'max_df': 0.95, 'min_df': 3, 'max_features': None}]

    def main(self):
        # phase 1: read data from files;
        self.read_data()

        # phase 2: model selection with some baseline params; 
        self.run_trials_select_model() # max for NaiveBayes Model with tfidfVectorizer

        # phase 3: fine-tune the best settings for some additional params
        self.fine_tune_params_on_selected_model_and_vectorizer()

        # phase 4: choose the best fraction for samples split;   
        self.tune_classfier_samples_size()

        # phase 5: run final train, with the optimal params and samples split ratios, as obtained in phase 3 & 4
        self.run_final_setup()

        # phase 6: check wrongly classified reviews
        self.check_content_misclassif()


if __name__ == "__main__":
    mrc = MoviesReviewClassifier()
    print 'start classifying:'
    mrc.main()