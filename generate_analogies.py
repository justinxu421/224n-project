from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from gensim.scripts.glove2word2vec import glove2word2vec
import pprint
import numpy as np

# load different word vectors into gensim model
def load_model(name):
	if name == 'fasttext':
		model = FastText.load_fasttext_format('./data/word_vectors/fast_text_biology.bin')
		return model

	if name == 'fasttext_pretrain':
		model = FastText.load_fasttext_format('wiki.en/wiki.en.bin')
		return model

	if name == 'biology':
		glove_file = './data/word_vectors/biology_vectors.txt'
	elif name == 'glove':
		glove_file = 'glove.6B.300d.txt'
	elif name == 'life':
		glove_file = './data/word_vectors/life_vectors.txt'

	tmp_file = get_tmpfile("pretrained_word2vec.txt")
	_ = glove2word2vec(glove_file, tmp_file)
	model = KeyedVectors.load_word2vec_format(tmp_file)
	return model

def get_most_similar_words(model, word1, word2, word3):
	scores = model.most_similar(positive=[word2, word3], negative=[word1])
	return [word for word, score in scores], [score for word,score in scores]

def get_successful_analogies(model, analogies):
	successes = []
	correct = []
	scores = []
	for analogy in analogies:
		word1, word2, word3, word4 = analogy.split()
		try:
			words, sim_scores = get_most_similar_words(model, word1, word2, word3)
			if word4 in words:
				successes.append(1)
				correct.append(analogy)
				scores.append(sim_scores)
			else:
				successes.append(0)
		# skip if out of vocabulary
		except:
			pass

	return successes, correct, scores

with open('data/analogy_data/unigram_analogy.txt') as f:
	analogies = f.readlines()

	# read in word vectors from defined wordvectors
	model = load_model('fasttext_pretrain')

	successes, correct,scores = get_successful_analogies(model, analogies)

	print(sum(successes))
	print(len(successes))
	print(np.mean(np.array(scores)))
	pprint.pprint(correct)
