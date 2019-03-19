from allennlp.modules.elmo import Elmo, batch_to_ids
from lstm_baseline import LstmModel
import torch
import itertools
import torch.optim as optim
import numpy as np
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.modules.attention import LinearAttention, BilinearAttention, DotProductAttention
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
# from Seq2SeqAnalogyModel import Seq2SeqAnalogyModel
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.predictors import SimpleSeq2SeqPredictor
from AnalogyDatasetReader import AnalogyDatasetReader
from predictors import SentenceClassifierPredictor
from typing import Iterator, List, Dict

#Modified based on https://github.com/allenai/allennlp and tutorial on RealWorldNLP

DATA_ROOT='../data/analogy_data'
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo_embedding_dim = 1024
hidden_dim = 1024
USE_GPU = torch.cuda.is_available()

def predict():
	with open("./model_seq2seq.th", 'rb') as f:
		model2.load_state_dict(torch.load(f))
	
	predictor2 = SentenceClassifierPredictor(model2, dataset_reader=reader)
	with open('test_seq2seq.txt', 'w+') as f:
		for analogy_test in test_dataset:
			logits = predictor2.predict_instance(analogy_test)['logits']
			label_id = np.argmax(logits)
			label_predict = model2.vocab.get_token_from_index(label_id, 'labels')
			f.write(label_predict + "\n")

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
	elmo_token_indexer = ELMoTokenCharactersIndexer()

	reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=WordTokenizer(),
        source_token_indexers={'tokens': elmo_token_indexer},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})

	
	train_dataset, test_dataset, dev_dataset = (reader.read(DATA_ROOT + "/" + fname) for fname in ["train_all_seq.txt", "test_all_seq.txt", "val_all_seq.txt"])

	vocab = Vocabulary.from_instances(train_dataset + dev_dataset + test_dataset,
                                      min_count={'tokens': 1, 'target_tokens': 1})

	# en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	#                              embedding_dim=256)
	# en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             # embedding_dim=elmo_embedding_dim)
	#elmo_embedder = Elmo(options_file, weight_file, 2, dropout=0.5)
	elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
	# word_embeddings = BasicTextFieldEmbedder({'tokens': elmo_embedder})
	# en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             # embedding_dim=256)
	source_embedder = BasicTextFieldEmbedder({"tokens": elmo_embedder})

	#Initializing the model
	max_decoding_steps = 20  
	encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(elmo_embedding_dim, hidden_dim, batch_first=True))

	# encoder = StackedSelfAttentionEncoder(input_dim=elmo_embedding_dim, hidden_dim=hidden_dim, projection_dim=128, feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)
	attention = DotProductAttention()

	model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=elmo_embedding_dim,
                          target_namespace='target_tokens',
	                          attention=attention,
                          beam_size=8,
                          use_bleu=True)

	if USE_GPU: model.cuda()
	else: model

	# Training the model 
	optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
	iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])
	iterator.index_with(vocab)

	trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=dev_dataset,
                  patience=10,
                  num_epochs=1, cuda_device=0 if USE_GPU else -1)

	for i in range(20):
		print('Epoch: {}'.format(i))
		trainer.train()

		predictor = SimpleSeq2SeqPredictor(model, reader)

		for instance in itertools.islice(dev_dataset, 10):
			print('SOURCE:', instance.fields['source_tokens'].tokens)
			print('GOLD:', instance.fields['target_tokens'].tokens)
			print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])

	#Saving the model
	with open("model_seq2seq.th", 'wb') as f:
		torch.save(model.state_dict(), f)

	vocab.save_to_files("vocabulary_seq2seq")
	predictor = SimpleSeq2SeqPredictor(model, reader)
	with open('predict_seq2seq.txt', 'w+') as f:
		for instance in itertools.islice(test_dataset, 10):
			preds = predictor.predict_instance(instance)['predicted_tokens']
			f.write(" ".join(preds) + "\n")
	

if __name__ == '__main__':
	main()
	# predict()
	# eval_predictions("test_seq2seq.txt", DATA_ROOT + "/" + "test_sm_seq2seq.txt")

