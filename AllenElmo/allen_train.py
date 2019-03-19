from allennlp.modules.elmo import Elmo, batch_to_ids
from lstm_baseline import LstmModel
import torch
import torch.optim as optim
import numpy as np
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from AnalogyDatasetReader import AnalogyDatasetReader
from predictors import SentenceClassifierPredictor
from typing import Iterator, List, Dict

#Modified based on https://github.com/allenai/allennlp and tutorial on RealWorldNLP

DATA_ROOT='../data/analogy_data'
USE_GPU = torch.cuda.is_available()
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo_embedding_dim = 1024
hidden_dim = 128


def predict():
	elmo_token_indexer = ELMoTokenCharactersIndexer()

	reader = AnalogyDatasetReader(token_indexers={'tokens':elmo_token_indexer})

	train_dataset, test_dataset, dev_dataset = (reader.read(DATA_ROOT + "/" + fname) for fname in ["train_all.txt", "test_all.txt", "val_all.txt"])

	# elmo_embedder = Elmo(options_file, weight_file, 2, dropout=0.5)
	elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
	
	word_embeddings = BasicTextFieldEmbedder({'tokens': elmo_embedder})
	lstm_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(elmo_embedding_dim, hidden_dim, batch_first=True, bidirectional=True))

	vocab2 = Vocabulary.from_files("./vocabulary")
	model2 = LstmModel(word_embeddings, lstm_encoder, vocab2)

	if USE_GPU: model2.cuda()
	else: model2

	with open("./model.th", 'rb') as f:
		model2.load_state_dict(torch.load(f))
	
	predictor2 = SentenceClassifierPredictor(model2, dataset_reader=reader)
	with open('test.txt', 'w+') as f:
		top_10_words_list = []
		for analogy_test in test_dataset:
			logits = predictor2.predict_instance(analogy_test)['logits']
			label_id = np.argmax(logits)
			label_predict = model2.vocab.get_token_from_index(label_id, 'labels')

			top_10_ids = np.argsort(logits)[-10:]
			top_10_words = [model2.vocab.get_token_from_index(id, 'labels') for id in top_10_ids]
			top_10_words_list.append(top_10_words)
			f.write(label_predict + "\n")

	top_10_words_list = np.array(top_10_words_list)
	print(top_10_words_list.shape)
	np.save('elmo_top_10_words_list.npy', np.array(top_10_words_list))

def eval_predictions(predict_path, gold_path):
	lines_predict = []
	lines_gold_path = []
	print(predict_path)
	with open(predict_path, 'r') as f:
		lines_predict = f.readlines()
		lines_predict = [line.strip() for line in lines_predict]

	with open(gold_path, 'r') as f_gold:
		for line in f_gold:
			analogyList = line.lower().strip().split(' | ')[1:]
			lines_gold_path.append(analogyList[3].strip())

	num_correct = 0
	for line_predict, line_gold in zip(lines_predict, lines_gold_path):
		if line_predict == line_gold:
			num_correct += 1
	accuracy = num_correct/len(lines_gold_path)
	print("Accuracy: " + str(accuracy))
	print("Num correct: " + str(num_correct))


def main ():
	#Initlizing the embeddings (ELMO)
	elmo_token_indexer = ELMoTokenCharactersIndexer()

	reader = AnalogyDatasetReader(token_indexers={'tokens':elmo_token_indexer})

	train_dataset, test_dataset, dev_dataset = (reader.read(DATA_ROOT + "/" + fname) for fname in ["train_all.txt", "test_all.txt", "val_all.txt"])

	# elmo_embedder = Elmo(options_file, weight_file, 2, dropout=0.5)
	elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
	
	vocab = Vocabulary.from_instances(train_dataset + test_dataset + dev_dataset)
	word_embeddings = BasicTextFieldEmbedder({'tokens': elmo_embedder})
	#Initializing the model
	#takes the hidden state at the last time step of the LSTM for every layer as one single output
	lstm_encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(elmo_embedding_dim, hidden_dim, batch_first=True, bidirectional=True))
	model = LstmModel(word_embeddings, lstm_encoder, vocab)

	if USE_GPU: model.cuda()
	else: model

	# Training the model 
	optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
	iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])
	iterator.index_with(vocab)

	trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  patience=10,
                  cuda_device=0 if USE_GPU else -1,
                  num_epochs=20)

	trainer.train()

	#Saving the model
	with open("model.th", 'wb') as f:
		torch.save(model.state_dict(), f)

	vocab.save_to_files("vocabulary")
	

if __name__ == '__main__':
	# main()
	predict()
	eval_predictions("test.txt", DATA_ROOT + "/" + "test_all.txt")

