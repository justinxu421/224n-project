# TODO: finish this to analyze data
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from collections import defaultdict

DATA_ROOT='data/analogy_data'

def get_lines_gold_path(gold_path):
	lines_gold_path = []
	with open(gold_path, 'r') as f_gold:
		for line in f_gold:
			analogyList = line.lower().strip().split(' | ')[1:]
			lines_gold_path.append(analogyList[3].strip())
	return lines_gold_path	

def get_top_10_acc(n, top_10_list, lines_gold_path):
	correct, count = 0,0
	for i, (top_10, gold) in enumerate(zip(top_10_list, lines_gold_path)):
		# if i < 20:
		# 	print(top_10[10-n:], gold) 
		if gold in top_10[10-n:]:
			correct += 1
		count += 1
	return correct/count

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

def get_bleu(top_10_list, gold_path):
	idx_to_tuple, lines_dict = get_line_dict(gold_path)
	# get the bleu score
	correct, count = 0,0
	list_of_ref = []
	hypotheses = []

	sentence_bleus = []
	for idx, top_10 in enumerate(top_10_list):
		# split so that sentence is array of words
		references = [x.split(' ')for x in lines_dict[idx_to_tuple[idx]]]
		hypothesis = top_10[-1].split(' ')
		list_of_ref.append(references)
		hypotheses.append(hypothesis)

	return 100*corpus_bleu(list_of_ref, hypotheses, weights=[.5,.5,0,0])


def get_raw_acc(top_10_list, gold_path):
	idx_to_tuple, lines_dict = get_line_dict(gold_path)
	# get the raw acc
	correct, count = 0,0
	for idx, top_10 in enumerate(top_10_list):
		if top_10[-1] in lines_dict[idx_to_tuple[idx]]:
			correct += 1
		count += 1
	return correct/count
		
gold_path = DATA_ROOT + "/" + "test_all.txt"
paths = ['AllenElmo/elmo_top_10_words_list.npy', 'AllenBert/bert_top_10_words_list.npy', 'AllenBert/biobert/top_10_words_list.npy']

lines_gold_path = get_lines_gold_path(gold_path)
for path in paths:
	print(path)

	top_10_list = np.load(path)	
	for n in range(1,5):
		print('top {} accuracy: {}'.format(n, get_top_10_acc(n, top_10_list, lines_gold_path)))

	print('raw acc any match: {}'.format(get_raw_acc(top_10_list, gold_path)))
	print('corpus bleu: {}'.format(get_bleu(top_10_list, gold_path)))
	print()