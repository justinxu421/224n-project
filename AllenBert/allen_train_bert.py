from allennlp.modules.elmo import Elmo, batch_to_ids
from lstm_baseline import LstmModel
import torch
import torch.optim as optim
import numpy as np
from overrides import overrides
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.data.vocabulary import Vocabulary

# try to do bert 
from allennlp.data.token_indexers import PretrainedBertIndexer
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

from allennlp.modules.seq2vec_encoders.pytorch_seq2vec_wrapper import PytorchSeq2VecWrapper
# from allennlp.modules.token_embedders import ElmoTokenEmbedder
from BertAnalogyDatasetReader import BertAnalogyDatasetReader
from predictors import SentenceClassifierPredictor
from typing import Iterator, List, Dict
from config import Config

#Modified based on https://github.com/allenai/allennlp and tutorial on RealWorldNLP

DATA_ROOT='../data/analogy_data'
USE_GPU = torch.cuda.is_available()

config = Config(
    testing=True,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100, # necessary to limit memory usage
    max_vocab_size=100000,
)

def bert_tokenizer(s: str):
    return token_indexer.wordpiece_tokenizer(s)[:config.max_seq_len - 2]

def predict(vocab2):
	bert_token_indexer = PretrainedBertIndexer(
	    pretrained_model="bert-large-uncased",
	    max_pieces=config.max_seq_len,
	    do_lowercase=True,
	)
	reader = BertAnalogyDatasetReader(
		tokenizer=bert_tokenizer, 
		token_indexers={'tokens':bert_token_indexer}
	)	

	train_dataset, test_dataset, dev_dataset = (reader.read(DATA_ROOT + "/" + fname) for fname in ["train_all.txt", "test_all.txt", "val_all.txt"])

	bert_embedder = PretrainedBertEmbedder(
	         pretrained_model="bert-large-uncased",
	         top_layer_only=True, # conserve memory
	)
	word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
	                                                             # we'll be ignoring masks so we'll need to set this to True
	                                                            allow_unmatched_keys = True)

	BERT_DIM = word_embeddings.get_output_dim()
	class BertSentencePooler(Seq2VecEncoder):
	    def forward(self, embs: torch.tensor, 
	                mask: torch.tensor=None) -> torch.tensor:
	        # extract first token tensor
	        return embs[:, 0]
	    
	    @overrides
	    def get_output_dim(self) -> int:
	        return BERT_DIM
	    
	# vocab2 = Vocabulary.from_files("./bert_vocabulary")
	bert_encoder = BertSentencePooler(vocab2)
	model2 = LstmModel(word_embeddings, bert_encoder, vocab2)
	with open("./bert_model.th", 'rb') as f:
		model2.load_state_dict(torch.load(f))
	
	predictor2 = SentenceClassifierPredictor(model2, dataset_reader=reader)
	with open('bert_predictions.txt', 'w+') as f:
		top_10_words_list = []
		for analogy_test in test_dataset:
			logits = predictor2.predict_instance(analogy_test)['logits']
			label_id = np.argmax(logits)
			label_predict = model2.vocab.get_token_from_index(label_id, 'labels')

			top_10_ids = np.argsort(logits)[-10:]
			top_10_words = [model2.vocab.get_token_from_index(id, 'labels') for id in top_10_ids]
			top_10_words_list.append(top_10_words)
			f.write(label_predict + "\n")

	np.array(top_10_words_list).savetxt('bert_top_10_words_list.out')

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


def main():
	#Initlizing the embeddings (BERT)
	bert_token_indexer = PretrainedBertIndexer(
	    pretrained_model="bert-large-uncased",
	    max_pieces=config.max_seq_len,
	    do_lowercase=True,
	)
	reader = BertAnalogyDatasetReader(
		tokenizer=bert_tokenizer,
		token_indexers={'tokens':bert_token_indexer}
	)

	train_dataset, test_dataset, dev_dataset = (reader.read(DATA_ROOT + "/" + fname) for fname in ["train_all.txt", "test_all.txt", "val_all.txt"])
	
	vocab = Vocabulary.from_instances(train_dataset + test_dataset + dev_dataset)


	bert_embedder = PretrainedBertEmbedder(
	         pretrained_model="bert-large-uncased",
	         top_layer_only=True, # conserve memory
	)
	word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder},
	                                                             # we'll be ignoring masks so we'll need to set this to True
	                                                            allow_unmatched_keys = True)

	BERT_DIM = word_embeddings.get_output_dim()

	class BertSentencePooler(Seq2VecEncoder):
	    def forward(self, embs: torch.tensor, 
	                mask: torch.tensor=None) -> torch.tensor:
	        # extract first token tensor
	        return embs[:, 0]
	    
	    @overrides
	    def get_output_dim(self) -> int:
	        return BERT_DIM
	 
	#Initializing the model
	#takes the hidden state at the last time step of the LSTM for every layer as one single output
	bert_encoder = BertSentencePooler(vocab)

	model = LstmModel(word_embeddings, bert_encoder, vocab)
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
	with open("bert_model.th", 'wb') as f:
		torch.save(model.state_dict(), f)

	vocab.save_to_files("bert_vocabulary")
	return vocab

if __name__ == '__main__':
	vocab = main()
	predict(vocab)
	eval_predictions("bert_predictions.txt", DATA_ROOT + "/" + "test_all.txt")

