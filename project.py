import os
import nltk.classify.util
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.classify import NaiveBayesClassifier
import collections
from nltk.metrics import precision
from nltk.metrics import recall
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
import sys

class Corpus:    
    def __init__(self):
        self.imdbtestposdir = "aclImdb/test/pos/"
        self.imdbtestnegdir = "aclImdb/test/neg/"
        self.imdbtrainposdir = "aclImdb/train/pos/"
        self.imdbtrainnegdir = "aclImdb/train/neg/"

        self.imdbtestposcorpus = PlaintextCorpusReader(self.imdbtestposdir,'.*')
        self.imdbtestnegcorpus = PlaintextCorpusReader(self.imdbtestnegdir,'.*')
        self.imdbtrainposcorpus = PlaintextCorpusReader(self.imdbtrainposdir,'.*')
        self.imdbtrainnegcorpus = PlaintextCorpusReader(self.imdbtrainnegdir,'.*')
        
        self.test_7_corpus = PlaintextCorpusReader(self.imdbtestposdir,'[0-9]+_7\.txt')
        self.test_8_corpus = PlaintextCorpusReader(self.imdbtestposdir,'[0-9]+_8\.txt')
        self.test_9_corpus = PlaintextCorpusReader(self.imdbtestposdir,'[0-9]+_9\.txt')
        self.test_10_corpus = PlaintextCorpusReader(self.imdbtestposdir,'[0-9]+_10\.txt')
        self.test_1_corpus = PlaintextCorpusReader(self.imdbtestnegdir,'[0-9]+_1\.txt')
        self.test_2_corpus = PlaintextCorpusReader(self.imdbtestnegdir,'[0-9]+_2\.txt')
        self.test_3_corpus = PlaintextCorpusReader(self.imdbtestnegdir,'[0-9]+_3\.txt')
        self.test_4_corpus = PlaintextCorpusReader(self.imdbtestnegdir,'[0-9]+_4\.txt')
        
        self.train_7_corpus = PlaintextCorpusReader(self.imdbtrainposdir,'[0-9]+_7\.txt')
        self.train_8_corpus = PlaintextCorpusReader(self.imdbtrainposdir,'[0-9]+_8\.txt')
        self.train_9_corpus = PlaintextCorpusReader(self.imdbtrainposdir,'[0-9]+_9\.txt')
        self.train_10_corpus = PlaintextCorpusReader(self.imdbtrainposdir,'[0-9]+_10\.txt')
        self.train_1_corpus = PlaintextCorpusReader(self.imdbtrainnegdir,'[0-9]+_1\.txt')
        self.train_2_corpus = PlaintextCorpusReader(self.imdbtrainnegdir,'[0-9]+_2\.txt')
        self.train_3_corpus = PlaintextCorpusReader(self.imdbtrainnegdir,'[0-9]+_3\.txt')
        self.train_4_corpus = PlaintextCorpusReader(self.imdbtrainnegdir,'[0-9]+_4\.txt')             
     
class Classifier:    
    def __init__(self):
        print "Importing Corpus..."
        self.corpus = Corpus()
        print "Corpus Imported"
        print "Constructing best word dictionary from corpus..."
        self.bestwords = self.best_rating_words_set_count()
        print "Best word dictionary constructed"
        
    def neg_12_rating_classifier(self):
        print "training classifier"
        print "1"
        self.crps = self.corpus.train_1_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        print "2" 
        self.crps = self.corpus.train_2_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "1"
        self.crps = self.corpus.test_1_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        print "2" 
        self.crps = self.corpus.test_2_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '1 precision:', precision(self.refsets['1'], self.testsets['1'])
        print '1 recall:', recall(self.refsets['1'], self.testsets['1'])
        print
        print '2 precision:', precision(self.refsets['2'], self.testsets['2'])
        print '2 recall:', recall(self.refsets['2'], self.testsets['2'])
        print
    
    def neg_13_rating_classifier(self):
        print "training classifier"
        print "1"
        self.crps = self.corpus.train_1_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        print "3" 
        self.crps = self.corpus.train_3_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "1"
        self.crps = self.corpus.test_1_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        print "3" 
        self.crps = self.corpus.test_2_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '1 precision:', precision(self.refsets['1'], self.testsets['1'])
        print '1 recall:', recall(self.refsets['1'], self.testsets['1'])
        print
        print '3 precision:', precision(self.refsets['3'], self.testsets['3'])
        print '3 recall:', recall(self.refsets['3'], self.testsets['3'])
        print
    
    def neg_14_rating_classifier(self):
        print "training classifier"
        print "1"
        self.crps = self.corpus.train_1_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        print "4" 
        self.crps = self.corpus.train_4_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "1"
        self.crps = self.corpus.test_1_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        print "4" 
        self.crps = self.corpus.test_4_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '1 precision:', precision(self.refsets['1'], self.testsets['1'])
        print '1 recall:', recall(self.refsets['1'], self.testsets['1'])
        print
        print '4 precision:', precision(self.refsets['4'], self.testsets['4'])
        print '4 recall:', recall(self.refsets['4'], self.testsets['4'])
        print
    
    def neg_23_rating_classifier(self):
        print "training classifier"
        print "2"
        self.crps = self.corpus.train_2_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]
        print "3" 
        self.crps = self.corpus.train_3_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "2"
        self.crps = self.corpus.test_2_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]
        print "3" 
        self.crps = self.corpus.test_3_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '2 precision:', precision(self.refsets['2'], self.testsets['2'])
        print '2 recall:', recall(self.refsets['2'], self.testsets['2'])
        print
        print '3 precision:', precision(self.refsets['3'], self.testsets['3'])
        print '3 recall:', recall(self.refsets['3'], self.testsets['3'])
        print
    
    def neg_24_rating_classifier(self):
        print "training classifier"
        print "2"
        self.crps = self.corpus.train_2_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]
        print "4" 
        self.crps = self.corpus.train_4_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "2"
        self.crps = self.corpus.test_2_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]
        print "4" 
        self.crps = self.corpus.test_4_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '2 precision:', precision(self.refsets['2'], self.testsets['2'])
        print '2 recall:', recall(self.refsets['2'], self.testsets['2'])
        print
        print '4 precision:', precision(self.refsets['4'], self.testsets['4'])
        print '4 recall:', recall(self.refsets['4'], self.testsets['4'])
        print
     
    def neg_34_rating_classifier(self):
        print "training classifier"
        print "3"
        self.crps = self.corpus.train_3_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]
        print "4" 
        self.crps = self.corpus.train_4_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "3"
        self.crps = self.corpus.test_3_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]
        print "4" 
        self.crps = self.corpus.test_4_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '3 precision:', precision(self.refsets['3'], self.testsets['3'])
        print '3 recall:', recall(self.refsets['3'], self.testsets['3'])
        print
        print '4 precision:', precision(self.refsets['4'], self.testsets['4'])
        print '4 recall:', recall(self.refsets['4'], self.testsets['4'])
        print 
    
    def pos_78_rating_classifier(self):
        print "training classifier"
        print "7"
        self.crps = self.corpus.train_7_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        print "8" 
        self.crps = self.corpus.train_8_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "7"
        self.crps = self.corpus.test_7_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        print "8" 
        self.crps = self.corpus.test_8_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '7 precision:', precision(self.refsets['7'], self.testsets['7'])
        print '7 recall:', recall(self.refsets['7'], self.testsets['7'])
        print
        print '8 precision:', precision(self.refsets['8'], self.testsets['8'])
        print '8 recall:', recall(self.refsets['8'], self.testsets['8'])
        print
    
    def pos_79_rating_classifier(self):
        print "training classifier"
        print "7"
        self.crps = self.corpus.train_7_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        print "9" 
        self.crps = self.corpus.train_9_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "7"
        self.crps = self.corpus.test_7_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        print "9" 
        self.crps = self.corpus.test_9_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '7 precision:', precision(self.refsets['7'], self.testsets['7'])
        print '7 recall:', recall(self.refsets['7'], self.testsets['7'])
        print
        print '9 precision:', precision(self.refsets['9'], self.testsets['9'])
        print '9 recall:', recall(self.refsets['9'], self.testsets['9'])
        print
    
    def pos_710_rating_classifier(self):
        print "training classifier"
        print "7"
        self.crps = self.corpus.train_7_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        print "10" 
        self.crps = self.corpus.train_10_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "7"
        self.crps = self.corpus.test_7_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        print "10" 
        self.crps = self.corpus.test_10_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '7 precision:', precision(self.refsets['7'], self.testsets['7'])
        print '7 recall:', recall(self.refsets['7'], self.testsets['7'])
        print
        print '10 precision:', precision(self.refsets['10'], self.testsets['10'])
        print '10 recall:', recall(self.refsets['10'], self.testsets['10'])
        print
    
    def pos_89_rating_classifier(self):
        print "training classifier"
        print "8"
        self.crps = self.corpus.train_8_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]
        print "9" 
        self.crps = self.corpus.train_9_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "8"
        self.crps = self.corpus.test_8_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]
        print "9" 
        self.crps = self.corpus.test_9_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '8 precision:', precision(self.refsets['8'], self.testsets['8'])
        print '8 recall:', recall(self.refsets['8'], self.testsets['8'])
        print
        print '9 precision:', precision(self.refsets['9'], self.testsets['9'])
        print '9 recall:', recall(self.refsets['9'], self.testsets['9'])
        print
    
    def pos_810_rating_classifier(self):
        print "training classifier"
        print "8"
        self.crps = self.corpus.train_8_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]
        print "10" 
        self.crps = self.corpus.train_10_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "8"
        self.crps = self.corpus.test_8_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]
        print "10" 
        self.crps = self.corpus.test_10_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '8 precision:', precision(self.refsets['8'], self.testsets['8'])
        print '8 recall:', recall(self.refsets['8'], self.testsets['8'])
        print
        print '10 precision:', precision(self.refsets['10'], self.testsets['10'])
        print '10 recall:', recall(self.refsets['10'], self.testsets['10'])
        print
     
    def pos_910_rating_classifier(self):
        print "training classifier"
        print "9"
        self.crps = self.corpus.train_9_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]
        print "10" 
        self.crps = self.corpus.train_10_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids]  
             
        print 'train on %d instances' % (len(self.trainfeat))    
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        print "9"
        self.crps = self.corpus.test_9_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]
        print "10" 
        self.crps = self.corpus.test_10_corpus
        self.fileids = self.crps.fileids() 
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids] 
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features()
        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print '9 precision:', precision(self.refsets['9'], self.testsets['9'])
        print '9 recall:', recall(self.refsets['9'], self.testsets['9'])
        print
        print '10 precision:', precision(self.refsets['10'], self.testsets['10'])
        print '10 recall:', recall(self.refsets['10'], self.testsets['10'])
        print
            
    def sentiment_classifier(self):
        
        self.crps = self.corpus.imdbtrainposcorpus
        self.fileids = self.crps.fileids()
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'pos') for f in self.fileids] 
        self.crps = self.corpus.imdbtrainnegcorpus
        self.fileids = self.crps.fileids()
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'neg') for f in self.fileids]
        
        print 'train on %d instances' % (len(self.trainfeat))
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        
        self.trainfeat = None
        self.crps = self.corpus.imdbtestposcorpus
        self.fileids = self.crps.fileids()
        self.testfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'pos') for f in self.fileids] 
        self.crps = self.corpus.imdbtestnegcorpus
        self.fileids = self.crps.fileids()
        self.testfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'neg') for f in self.fileids]
        
        print 'test on %d instances' % (len(self.testfeat))
        
        print 'accuracy:', nltk.classify.util.accuracy(self.classifier, self.testfeat)
        self.classifier.show_most_informative_features() 

        
        self.refsets = collections.defaultdict(set)
        self.testsets = collections.defaultdict(set)
         
        for i, (feats, label) in enumerate(self.testfeat):
            self.refsets[label].add(i)
            self.observed = self.classifier.classify(feats)
            self.testsets[self.observed].add(i)
        self.testfeat = None 
        print 
        print 'pos precision:', precision(self.refsets['pos'], self.testsets['pos'])
        print 'pos recall:', recall(self.refsets['pos'], self.testsets['pos'])
        print
        print 'neg precision:', precision(self.refsets['neg'], self.testsets['neg'])
        print 'neg recall:', recall(self.refsets['neg'], self.testsets['neg'])
        print 
    
    def retrieve_neg_12_rating_classifier(self):
        self.crps = self.corpus.train_1_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        self.crps = self.corpus.train_2_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]  
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
    
    def retrieve_neg_13_rating_classifier(self):
        self.crps = self.corpus.train_1_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        self.crps = self.corpus.train_3_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]  
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
    
    def retrieve_neg_14_rating_classifier(self):
        self.crps = self.corpus.train_1_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'1') for f in self.fileids]
        self.crps = self.corpus.train_4_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids]  
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier       
     
    def retrieve_neg_23_rating_classifier(self):
        self.crps = self.corpus.train_2_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]
        self.crps = self.corpus.train_3_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
    
    def retrieve_neg_24_rating_classifier(self):
        self.crps = self.corpus.train_2_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'2') for f in self.fileids]
        self.crps = self.corpus.train_4_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
     
    def retrieve_neg_34_rating_classifier(self):
        self.crps = self.corpus.train_3_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'3') for f in self.fileids]
        self.crps = self.corpus.train_4_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'4') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier 
    
    def retrieve_pos_78_rating_classifier(self):
        self.crps = self.corpus.train_7_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        self.crps = self.corpus.train_8_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
    
    def retrieve_pos_79_rating_classifier(self):
        self.crps = self.corpus.train_7_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        self.crps = self.corpus.train_9_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]  
 
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
    
    def retrieve_pos_710_rating_classifier(self):
        self.crps = self.corpus.train_7_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'7') for f in self.fileids]
        self.crps = self.corpus.train_10_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
    
    def retrieve_pos_89_rating_classifier(self):
        self.crps = self.corpus.train_8_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]
        self.crps = self.corpus.train_9_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier      
    
    def retrieve_pos_810_rating_classifier(self):
        self.crps = self.corpus.train_8_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'8') for f in self.fileids]
        self.crps = self.corpus.train_10_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids]  

        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier
        
     
    def retrieve_pos_910_rating_classifier(self):
        self.crps = self.corpus.train_9_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'9') for f in self.fileids]
        self.crps = self.corpus.train_10_corpus
        self.fileids = self.crps.fileids() 
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'10') for f in self.fileids]  
    
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None
        return self.classifier    
     
    def retrieve_sentiment_classifier(self):
        self.crps = self.corpus.imdbtrainposcorpus
        self.fileids = self.crps.fileids()
        self.trainfeat = [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'pos') for f in self.fileids] 
        self.crps = self.corpus.imdbtrainnegcorpus
        self.fileids = self.crps.fileids()
        self.trainfeat += [(self.best_bigram_word_feats(self.crps.words(fileids=[f])),'neg') for f in self.fileids]
        
        self.classifier = NaiveBayesClassifier.train(self.trainfeat)
        self.trainfeat = None 
        return self.classifier
         
    def classify_input(self,words):
        print "Classifying Input"
        self.classifier = self.retrieve_sentiment_classifier()
        self.input_words = self.best_bigram_word_feats(words)
        self.classification = self.classifier.classify(self.input_words)
        print "The review is primarily a ",self.classification
        if(self.classification == "pos"):
            count = [0,0,0,0]
            self.classifier = self.retrieve_pos_78_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-7] += 1
            
            self.classifier = self.retrieve_pos_79_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-7] += 1
            
            self.classifier = self.retrieve_pos_710_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-7] += 1
            
            self.classifier = self.retrieve_pos_89_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-7] += 1
            
            self.classifier = self.retrieve_pos_810_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-7] += 1
            
            self.classifier = self.retrieve_pos_910_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-7] += 1
            
            if(count[0] >= count[1] and count[0] >= count[2] and count[0] >= count[3]):
                print "Classification: 7"
            elif(count[1] >= count[0] and count[1] >= count[2] and count[1] >= count[3]):
                print "Classification: 8"
            elif(count[2] >= count[0] and count[2] >= count[1] and count[2] >= count[3]):
                print "Classification: 9"
            else:
                print "Classification: 10"            
        elif(self.classification == "neg"):
            count = [0,0,0,0]
            self.classifier = self.retrieve_neg_12_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-1] += 1
            
            self.classifier = self.retrieve_neg_13_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-1] += 1
            
            self.classifier = self.retrieve_neg_14_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-1] += 1
            
            self.classifier = self.retrieve_neg_23_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-1] += 1
            
            self.classifier = self.retrieve_neg_24_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-1] += 1
            
            self.classifier = self.retrieve_neg_34_rating_classifier()
            self.classification = self.classifier.classify(self.input_words)
            count[int(self.classification)-1] += 1
            
            if(count[0] >= count[1] and count[0] >= count[2] and count[0] >= count[3]):
                print "Classification: 1"
            elif(count[1] >= count[0] and count[1] >= count[2] and count[1] >= count[3]):
                print "Classification: 2"
            elif(count[2] >= count[0] and count[2] >= count[1] and count[2] >= count[3]):
                print "Classification: 3"
            else:
                print "Classification: 4"
        
    def word_feats(self,words):
        return dict([(word,True) for word in words])  
      
    def best_word_feats(self,words):
        self.stopset = set(stopwords.words('english'))
        return dict([(word,True) for word in words if word in self.bestwords and word not in self.stopset])  
    
    def stopword_filtered_word_feats(self,words):
        self.stopset = set(stopwords.words('english'))
        return dict([(word,True) for word in words if word not in self.stopset])  
                     
    def best_bigram_word_feats(self,words,score_fn=BigramAssocMeasures.chi_sq,n=500):
        self.bigram_finder = BigramCollocationFinder.from_words(words)
        self.bigrams = self.bigram_finder.nbest(score_fn, n)
        self.d = dict([(bigram, True) for bigram in self.bigrams])
        self.d.update(self.best_word_feats(words))
        return self.d
        
    def best_words_set_chi_sq(self):
        self.word_fd = FreqDist()
        self.label_word_fd = ConditionalFreqDist()
        
        for word in self.corpus.imdbtestposcorpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['pos'][word.lower()]+=1
            
        for word in self.corpus.imdbtestnegcorpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['neg'][word.lower()]+=1
            
        for word in self.corpus.imdbtrainposcorpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['pos'][word.lower()]+=1
            
        for word in self.corpus.imdbtrainnegcorpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['neg'][word.lower()]+=1 
            
        self.pos_word_count = self.label_word_fd['pos'].N()
        self.neg_word_count = self.label_word_fd['neg'].N()
        self.total_word_count = self.pos_word_count + self.neg_word_count
        
        self.word_scores = {}
        
        for word,freq in self.word_fd.iteritems():
            self.pos_score = BigramAssocMeasures.chi_sq(self.label_word_fd['pos'][word],(freq, self.pos_word_count), self.total_word_count)
            self.neg_score = BigramAssocMeasures.chi_sq(self.label_word_fd['neg'][word],(freq, self.neg_word_count), self.total_word_count)
            self.word_scores[word] = self.pos_score + self.neg_score
            
        self.best = sorted(self.word_scores.iteritems(), key=lambda (w,s): s, reverse=True)[:10000]
        self.bestwords = set([w for w, s in self.best]) 
        return self.bestwords                    
        
    def best_rating_words_set_count(self):
        self.word_fd = FreqDist()
        self.label_word_fd = ConditionalFreqDist()
        
        for word in self.corpus.test_1_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['1'][word.lower()]+=1
            
        for word in self.corpus.test_2_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['2'][word.lower()]+=1
            
        for word in self.corpus.test_3_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['3'][word.lower()]+=1
            
        for word in self.corpus.test_4_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['4'][word.lower()]+=1 
        
        for word in self.corpus.test_7_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['7'][word.lower()]+=1
            
        for word in self.corpus.test_8_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['8'][word.lower()]+=1
            
        for word in self.corpus.test_9_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['9'][word.lower()]+=1
            
        for word in self.corpus.test_10_corpus.words():
            self.word_fd[word.lower()]+=1
            self.label_word_fd['10'][word.lower()]+=1
            
        self._1_word_count = self.label_word_fd['1'].N()
        self._2_word_count = self.label_word_fd['2'].N()
        self._3_word_count = self.label_word_fd['3'].N()
        self._4_word_count = self.label_word_fd['4'].N()
        self._7_word_count = self.label_word_fd['5'].N()
        self._8_word_count = self.label_word_fd['6'].N()
        self._9_word_count = self.label_word_fd['7'].N()
        self._10_word_count = self.label_word_fd['8'].N()
        self.total_word_count = self._1_word_count + self._2_word_count + self._3_word_count + self._4_word_count + self._7_word_count + self._8_word_count + self._9_word_count + self._10_word_count
        self.pos_word_count = self._7_word_count + self._8_word_count + self._9_word_count + self._10_word_count
        self.neg_word_count = self._1_word_count + self._2_word_count + self._3_word_count + self._4_word_count
        self.word_scores = {}
        
        for word,freq in self.word_fd.iteritems():
            self.pos_score = self.label_word_fd['7'][word] + self.label_word_fd['8'][word] + self.label_word_fd['9'][word] + self.label_word_fd['10'][word]
            self.neg_score = self.label_word_fd['1'][word] + self.label_word_fd['2'][word] + self.label_word_fd['3'][word] + self.label_word_fd['4'][word]
            
            if(self.pos_score == 0):
                self.word_scores[word] = self.neg_score    
            elif(self.neg_score == 0):
                self.word_scores[word] = self.pos_score
            elif(self.pos_score > self.neg_score):
                self.word_scores[word] = self.pos_score / self.neg_score
            else:
                self.word_scores[word] = self.neg_score / self.pos_score    
                       
        self.best = []
        for word,value in self.word_scores.iteritems():
            if(value >= 2):
                self.best.append(word)
        self.stopset = set(stopwords.words('english'))
        self.bestwords = set([w for w in self.best]) 
        
        return self.bestwords    
              
        
print "Movie Rating Classifiers"
classifier = Classifier()
#classifier.neg_12_rating_classifier()
#classifier.neg_13_rating_classifier()
#classifier.neg_14_rating_classifier()
#classifier.neg_23_rating_classifier()
#classifier.neg_24_rating_classifier()
#classifier.neg_34_rating_classifier()
#classifier.pos_78_rating_classifier()
#classifier.pos_79_rating_classifier()
#classifier.pos_710_rating_classifier()
#classifier.pos_89_rating_classifier()
#classifier.pos_810_rating_classifier()
#classifier.pos_910_rating_classifier()
#classifier.sentiment_classifier()
i=1
while(i==1):
    print "1.Classify 2.Exit\nEnter:"
    i = int(raw_input())
    if(i==1):
        user_input = raw_input("Enter the review you want to classify:")
        classifier.classify_input(user_input.split()) 
    else:
        print "Bye"                                   
