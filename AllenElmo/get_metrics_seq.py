# TODO: finish this to analyze data
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import defaultdict

DATA_ROOT='../data/analogy_data'


def get_line_dict(gold_path):
	# map from (a,b,c) -> list of d
	lines_dict = defaultdict(list)
	# map the current index to the tuple key for lines_dict
	idx_to_tuple = {}
	with open(gold_path, 'r') as f_gold:
		for idx, line in enumerate(f_gold):
			a,b,c,d = line.lower().strip().split(' | ')[1:]
			key = (a.strip(), b.strip(), c.strip())
			lines_dict[key].append(d.strip())
			idx_to_tuple[idx] = key
	return idx_to_tuple, lines_dict

def get_bleu(top_sentences, gold_path):
	idx_to_tuple, lines_dict = get_line_dict(gold_path)
	# get the bleu score
	correct, count = 0,0
	list_of_ref = []
	hypotheses = []

	sentence_bleus = []
	for idx, top_elem in enumerate(top_sentences):
		# split so that sentence is array of words
		references = [x.split(' ')for x in lines_dict[idx_to_tuple[idx]]]
		hypothesis = top_elem.split(' ')
		list_of_ref.append(references)
		hypotheses.append(hypothesis)

	return 100*corpus_bleu(list_of_ref, hypotheses, weights=[.5,.5,0,0])


def get_raw_acc(top_analogies, gold_path):
	idx_to_tuple, lines_dict = get_line_dict(gold_path)
	# get the raw acc
	correct, count = 0,0
	for idx, top_elem in enumerate(top_analogies):
		if top_elem in lines_dict[idx_to_tuple[idx]]:
			correct += 1
		count += 1
	return correct/count
		
gold_path = DATA_ROOT + "/" + "test_all.txt"
paths= ['test_seq2seq_notest.txt']

for path in paths:
	print(path)
	top_analogies = []
	with open(path, 'r') as f:
		lines_predict = f.readlines()
		top_analogies = [line.strip() for line in lines_predict]		
	print('raw acc any match: {}'.format(get_raw_acc(top_analogies, gold_path)))
	print('corpus bleu: {}'.format(get_bleu(top_analogies, gold_path)))
	print()
