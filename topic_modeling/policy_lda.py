import nltk
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from gensim.models import ldamodel
from gensim.models import phrases
import pyLDAvis.gensim
import logging
import json
import io

""" 
Gensim topic modelling for BCa based on 
https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/ 
https://markroxor.github.io/gensim/static/notebooks/lda_training_tips.html
"""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def parsedPolicySegments(policyFile):
	policyData = []
	with open(policyFile) as f:
		policySegments = json.load(f)
	
	for key,value in policySegments.iteritems():
		policyData.append(value)
	
	return policyData

def readTxtFile(fileName):
	policyData = []
        with io.open(fileName, 'r', encoding="utf-8") as f:
		for line in f:
			policyData.append(line)
	
	return policyData

def cleanDocs(posts):
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation) 
	lemma = WordNetLemmatizer()
	clean_docs = []
	bigram_docs = []

	for post in posts: 
	    stop_free = " ".join([i for i in post.lower().split() if i not in stop])
	    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
	    digit_free = [word for word in punc_free.split() if not word.isdigit() and len(word) > 2]
	    normalized = " ".join(lemma.lemmatize(word) for word in digit_free)
	    nouns = [word[0] for word in nltk.pos_tag(normalized.split()) if word[1][0] == 'N' or word[1][0] == 'VB'] 
	    clean_docs.append(nouns)

	bigram_transformer = phrases.Phrases(clean_docs)
	
	for doc in bigram_transformer[clean_docs]:
			bigram_docs.append(doc)

	return bigram_docs 

def buildLADModel(clean_docs, model_name):
	# Creating the term dictionary of our courpus, where every unique term is assigned an index.
	dictionary = corpora.Dictionary(clean_docs)
	dictionary.save(model_name + '.dict')

	doc_term_matrix = []

	for doc in clean_docs:
		# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
		doc_term_matrix.append(dictionary.doc2bow(doc))

	corpora.MmCorpus.serialize(model_name + '.mm', doc_term_matrix)

	lda_model = ldamodel.LdaModel(doc_term_matrix, num_topics=100, id2word = dictionary, passes=50)

	lda_model.save(model_name + '.model')

	#print(lda_model.print_topics(num_topics=3, num_words=3))

def loadModel(modelname):
	lda_model = ldamodel.LdaModel.load(modelname)
	csvfile = open(modelname + '.txt', 'wb')

	#print(lda_model.print_topics(num_topics=100, num_words=10))
	for topic in range(lda_model.num_topics):
		print('\n')
		csvfile.write("Topic: " + str(topic)) 
		for word in lda_model.show_topic(topic, topn=50):
			csvfile.write(word)

def visulaizeModel(corpusfile, dictionaryfile, modelfile, visfile):
	"""Displaying gensim topic models"""
    ## Load files from "gensim_modeling"
   	corpus = corpora.MmCorpus(corpusfile)
   	dictionary = corpora.Dictionary.load(dictionaryfile) # for pyLDAvis
   	myldamodel = ldamodel.LdaModel.load(modelfile)    

    ## Interactive visualisation
   	vis = pyLDAvis.gensim.prepare(myldamodel, corpus, dictionary)
   	pyLDAvis.save_html(vis, visfile)

#dataFolderPath = "/home/lahiru/Research/policy_analysis/data/usableprivacy/OptOutChoice-2017_v1.0/SegmentDict.json"
dataFolderPath = "/home/lahiru/Research/policy_analysis/data/gdpr_doc/CELEX_32016R0679_EN_TXT.txt"
policySegments = readTxtFile(dataFolderPath)
clean_docs = cleanDocs(policySegments)
modelname = 'topic_100_gdpr_only_nouns_n_verb_run_1'
buildLADModel(clean_docs, modelname)

#loadModel(modelname + '.model')
visulaizeModel(modelname + '.mm', modelname + '.dict', modelname + '.model', modelname + '.html')
