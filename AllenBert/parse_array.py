# TODO: finish this to analyze data
import numpy as np

DATA_ROOT='../data/analogy_data'

def get_top_10_acc(n, top_10_path, gold_path):
	top_10_list = np.load(top_10_path)

	lines_gold_path = []
	with open(gold_path, 'r') as f_gold:
		for line in f_gold:
			analogyList = line.lower().strip().split(' | ')[1:]
			lines_gold_path.append(analogyList[3].strip())

	correct, count = 0,0
	for i, (top_10, gold) in enumerate(zip(top_10_list, lines_gold_path)):
		if i < 20:
			print(top_10[10-n:], gold)
		if gold in top_10[10-n:]:
			correct += 1
		count += 1
	return correct/count
		
top_10_path = 'biobert/top_10_words_list.npy'
gold_path = DATA_ROOT + "/" + "test_all.txt"
n = 3 
print('top {} accuracy: {}'.format(n, get_top_10_acc(n, top_10_path, gold_path)))
