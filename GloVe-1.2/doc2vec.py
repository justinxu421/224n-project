from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import os.path
import pprint
import numpy as np



if not os.path.isfile("doc2vec_tokenized_sentences.txt"):
	#preprocessing if preprocessing is not done.
	base_sentences = open("life_biology_sentences.txt").read()

	base_sentences = base_sentences.split('.')

	tokenizer = RegexpTokenizer(r'\w+')
	base_sentences = [ tokenizer.tokenize(sent)  for sent in base_sentences]
	base_sentences = [  ' '.join([word.lower() for word in token_list if word.isalpha()]) + '.' for token_list in base_sentences]


	with open('doc2vec_tokenized_sentences.txt', 'w') as f:
		for sent in base_sentences:
			f.write(sent + '\n')
else: 
	base_sentences = open("doc2vec_tokenized_sentences.txt").read()
	base_sentences = base_sentences.split('.')

##### Original model training:
# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(base_sentences)]
# model = Doc2Vec(documents, vector_size=300, window=2, min_count=1, workers=4)
fname = get_tmpfile("doc2vec_analogies")
# model.save(fname)
model = Doc2Vec.load(fname)

# Testing most similarity
# --------------------
# pprint.pprint(model.most_similar(positive=['enzyme', 'granum'], negative=['enzyme-substrate complex']))
# pprint.pprint(model.most_similar(positive=['photon', 'phosphodiester bond'], negative=['visible light']))
# not deterministic 
test = 'asks questions to emphasize scientific inquiry each of which is answered in a major section of the chapter.'
infered = np.array(model.infer_vector(test.split(' ')))
# infered = np.array(model.infer_vector(['photon', 'phosphodiester bond']))
coses = []
for index,sentence in enumerate(base_sentences):
	doc = np.array(model.docvecs[index])
	cos_n = model.docvecs[index] @ infered
	cos_d = np.linalg.norm(doc) * np.linalg.norm(infered)
	cos_sim  = cos_n/cos_d
	coses.append(cos_sim)

print(len(coses))
argmax = np.argmax(coses)
print(np.max(coses))
print(argmax)
print(base_sentences[argmax])
