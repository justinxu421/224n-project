from nltk.tokenize import word_tokenize

import nltk
file_content = open("life_biology_sentences.txt").read()
tokens = nltk.word_tokenize(file_content)

with open('life_biology_sentences_tokenized', 'w') as f:
	f.write(' '.join(tokens))
