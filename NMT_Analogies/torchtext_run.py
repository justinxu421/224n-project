#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>

Usage:
    run.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    run.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 50]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 100]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""
import math
import sys
import pickle
import time


from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nmt_model import Hypothesis, NMT
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

import torch
import torch.nn.utils


#torchtext
from torchtext.data import TabularDataset,Field,Iterator
import torchtext
import spacy
#tensorboard 
from tensorboardX import SummaryWriter


def evaluate_ppl(model, dev_iter):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.
    report_examples = 0
    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for _ , data in enumerate(dev_iter):
            (src_sents, src_lengths) , (tgt_sents, _)  = data.abc, data.d
            loss = model(src_sents,src_lengths,tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict


            batch_size = src_sents.shape[1]
            report_examples += batch_size
        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    val_loss = cum_loss / report_examples
    return ppl, val_loss


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    """
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']



    #prefer to do our entire train,test,val split in the code itself as opposed to our previous script
    # remove these comments

    #data preprocessing for Qs and As.
    spacy_en = spacy.load('en')

    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]


    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True,include_lengths=True,init_token='<s>',eos_token='</s>')
    analogies_datafields = [ ("abc", TEXT), ("d", TEXT)]

    train, val, test = TabularDataset.splits(
                   path="data", # the root directory where the data lies
                   train='ngram_train.csv', validation="ngram_val.csv", test= 'ngram_test.csv',
                   format='csv',
                   skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                   fields=analogies_datafields)


    pretrained_vecs = torchtext.vocab.Vectors('../GloVe-1.2/life_vectors.txt')
    TEXT.build_vocab(vectors=pretrained_vecs) # specials=['<pad>', '<s>', '</s>']


    if args['--cuda'] == 'cpu':
        torch_text_device = -1
    else:
        torch_text_device = 0 


    training_iter, val_iter, test_iter = Iterator.splits(
            (train, val, test), sort_key=lambda x: len(x.abc),
            batch_sizes=(100, 20, 1), device=torch_text_device, sort_within_batch=True)


    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=TEXT.vocab)
    model.train() #sets training = True

    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)


    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')


    writer = SummaryWriter('logs')
    is_better_count = 0 #TODO: Remove this and debug the nonstopping part
    while True:
        epoch += 1
        
        for _ , data in enumerate(training_iter):
            (src_sents, src_lengths) , (tgt_sents, _)  = data.abc, data.d

            train_iter += 1

            optimizer.zero_grad()

            batch_size = src_sents.shape[1]

            example_losses = model(src_sents,src_lengths,tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                writer.add_scalar('Train/AvgLoss', report_loss/report_examples, epoch)
                writer.add_scalar('Train/AvgPPL', math.exp(report_loss / report_tgt_words), epoch)
    

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl, val_loss = evaluate_ppl(model, val_iter)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f, dev loss %f' % (train_iter, dev_ppl,val_loss), file=sys.stderr)
                writer.add_scalar('Val/AvgPPL', dev_ppl, epoch)
                writer.add_scalar('Val/AvgLoss', val_loss,epoch)
                
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                print(hist_valid_scores)
                print(valid_metric)
                hist_valid_scores.append(valid_metric)
       
                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
                    is_better_count = is_better_count +  1
                    print(is_better_count)
                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                    if is_better_count > 3:
                        print('reached maximum number of epochs!', file=sys.stderr)
                        writer.close()
                        exit(0)

                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    writer.close()
                    exit(0)


def decode(args: Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """
    spacy_en = spacy.load('en')

    def tokenizer(text): # create a tokenizer function
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True,include_lengths=True,init_token='<s>',eos_token='</s>')
    analogies_datafields = [ ("abc", TEXT), ("d", TEXT)]

    train, val, test = TabularDataset.splits(
                   path="data", # the root directory where the data lies
                   train='ngram_train.csv', validation="ngram_val.csv", test= 'ngram_test.csv',
                   format='csv',
                   skip_header=False, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                   fields=analogies_datafields)

    pretrained_vecs = torchtext.vocab.Vectors('../GloVe-1.2/life_vectors.txt')
    TEXT.build_vocab(vectors=pretrained_vecs) # specials=['<pad>', '<s>', '</s>']


    if args['--cuda'] == 'cpu':
        torch_text_device = -1
    else:
        torch_text_device = 0 


    training_iter, val_iter, test_iter = Iterator.splits(
            (train, val, test), sort_key=lambda x: len(x.abc),
            batch_sizes=(100, 20, 1), device=torch_text_device, sort_within_batch=True)



    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_iter,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)

        #accuracy (unigrams)
        perfectly_correct = 0
        for index,hyp in enumerate(top_hypotheses):
            if hyp.value[0] == test_data_tgt[index][1]:
                perfectly_correct += 1
        print('Ignore accuracy for non unigrams')
        print('Accuracy: {}'.format(perfectly_correct / len(test_data_tgt)), file=sys.stderr)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model: NMT, test_iter, beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for _ , data in enumerate(test_iter):
            print(data)
            (src_sents, src_lengths) , (_ , _)  = data.abc, data.d
            example_hyps = model.beam_search(src_sents, src_lengths, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():
    """ Main func.
    """
    args = docopt(__doc__)
    print('Epochs:' + args['--max-epoch'])

    # Check pytorch version
    assert(torch.__version__ == "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
