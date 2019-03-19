#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-a=./data/train.es --train-tgt=./data/train.en --dev-src=./data/dev.es --dev-tgt=./data/dev.en --vocab=vocab.json --cuda --no-char-decoder  
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./data/test.es ./data/test.en outputs/test_outputs.txt --cuda --no-char-decoder  
elif [ "$1" = "train_local" ]; then
	python run.py train --train-a=./data/ngram_local_train.a --train-b=./data/ngram_local_train.b --train-c=./data/ngram_local_train.c --train-tgt=./data/ngram_local_train.d --dev-a=./data/ngram_local_val.a --dev-b=./data/ngram_local_val.b --dev-c=./data/ngram_local_val.c --dev-tgt=./data/ngram_local_val.d --vocab=vocab.json --no-char-decoder   
elif [ "$1" = "test_local" ]; then
    python run.py decode model.bin ./data/ngram_local_test.a ./data/ngram_local_test.b  ./data/ngram_local_test.c ./data/ngram_local_test.d outputs/ngram_local_char.txt
elif [ "$1" = "test_local_unigram_train" ]; then
    python run.py decode model.bin ./data/unigram_train.abc ./data/unigram_train.d outputs/test_unigram_train.txt
elif [ "$1" = "test_local_ngram_train" ]; then
    python run.py decode model.bin ./data/ngram_train.abc ./data/ngram_train.d outputs/test_ngram_train.txt
elif [ "$1" = "vocab" ]; then
	python vocab.py --train-src=./data/train.es --train-tgt=./data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
