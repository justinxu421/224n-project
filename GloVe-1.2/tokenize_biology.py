from nltk.tokenize import word_tokenize

# import nltk
from nltk.tokenize import RegexpTokenizer
file_content = open("life_biology_sentences.txt").read()

tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(file_content)
tokens = [word.lower() for word in tokens if word.isalpha()]

with open('life_biology_sentences_tokenized', 'w') as f:
	f.write(' '.join(tokens))
