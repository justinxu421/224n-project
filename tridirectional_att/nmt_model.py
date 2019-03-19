#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_char_embeddings import ModelCharEmbeddings
from model_word_embeddings import ModelWordEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


#Highway
from highway import Highway
from char_decoder import CharDecoder
class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab,weights,no_char_decoder=False, dropout_rate=0.2,):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()

        self.model_word_embeddings = ModelWordEmbeddings(embed_size, vocab,weights)

        self.model_char_embeddings_source = ModelCharEmbeddings(50, vocab.src)
        self.model_char_embeddings_target = ModelCharEmbeddings(50, vocab.src)

        # we set the embed_size = 2 * embed_size
        self.d = embed_size + 50
        # hidden_size = embed_size + 50


        self.highway = Highway( self.d)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.encoder = None 
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None


        ### YOUR CODE HERE (~8 Lines)
        ### TODO - Initialize the following variables:
        ###     self.encoder (Bidirectional LSTM with bias)
        ###     self.decoder (LSTM Cell with bias)
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        ###     self.dropout (Dropout Layer)
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        # print(embed_size + hidden_size)

        self.encoder_a = nn.LSTM(self.d,hidden_size,bidirectional=True, bias=True) #changed bias=True
        self.encoder_b = nn.LSTM(self.d,hidden_size,bidirectional=True, bias=True) #changed bias=True
        self.encoder_c= nn.LSTM(self.d,hidden_size,bidirectional=True, bias=True) #changed bias=True

        self.decoder = nn.LSTMCell((embed_size+hidden_size), 3*hidden_size, bias=True)

        self.att_projection = nn.Linear(10*hidden_size, 3*hidden_size, bias=False) 
        self.combined_output_projection = nn.Linear(13*hidden_size, hidden_size, bias=False)
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.src), bias=False) 
        self.dropout =  nn.Dropout(p=self.dropout_rate)


        self.ha_projection  = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.ca_projection  = nn.Linear(2*hidden_size, hidden_size, bias=False)

        self.hb_projection  = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.cb_projection  = nn.Linear(2*hidden_size, hidden_size, bias=False)

        self.hc_projection  = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.cc_projection  = nn.Linear(2*hidden_size, hidden_size, bias=False)

        #tri-attention
        self.Wab = nn.Linear(6*hidden_size, 1, bias=False)
        self.Wac = nn.Linear(6*hidden_size, 1, bias=False)
        self.Wbc = nn.Linear(6*hidden_size, 1, bias=False)

        if not no_char_decoder:
           self.charDecoder = CharDecoder(hidden_size, target_vocab=vocab.src) 
        else:
           self.charDecoder = None
        ### END YOUR CODE



    def tri_dir_att(self,encoded_a: torch.Tensor,encoded_b: torch.Tensor, encoded_c: torch.Tensor) -> torch.tensor:
        # encoded_a -> (b,max_len_a, 2h)
        # encoded_b -> (b,max_len_b, 2h)
        # encoded_c -> (b,max_len_c, 2h)
        batch_size = encoded_a.shape[0]
        len_a = encoded_a.shape[1]
        len_b = encoded_b.shape[1]
        len_c = encoded_c.shape[1]
        h2 = encoded_a.shape[2]

        #first a2b and b2a
        shape = (batch_size,len_a,len_b,h2)
        a_expanded = encoded_a.unsqueeze(2) # (b, len_a,1, 2h)
        a_expanded = a_expanded.expand(shape) # (b, len_a,len_b, 2h)
        b_expanded = encoded_b.unsqueeze(1) # (b,1, len_b, 2h)
        b_expanded = b_expanded.expand(shape) # (b, len_a, len_b, 2h)

        a_elmwise_mul_b = torch.mul(a_expanded, b_expanded) # (b, len_a, len_b, 2h)
        cat_data = torch.cat((a_expanded, b_expanded, a_elmwise_mul_b), 3) # (b, len_a,len_b, 6h), [h;u;h◦u]
        S = self.Wab(cat_data).view(batch_size, len_a, len_b) # (b, len_a, len_b)

        # a2b
        a2b = torch.bmm(F.softmax(S, dim=-1), encoded_b) # (b, len_a, 2h) = bmm( (b, len_a,len_b), (b, len_b, 2h) )
        # b2a
        # b: attention weights on the context
        b = F.softmax(torch.max(S, 2)[0], dim=-1) # (b, len_b)
        b2a = torch.bmm(b.unsqueeze(1), encoded_a) # (b, 1, 2d) = bmm( (b, 1, len_b), (b, len_b, 2d) )
        b2a = b2a.repeat(1, len_b, 1) # (b, len_b, 2d), tiled T times

        #next a2c and c2a
        shape = (batch_size,len_a,len_c,h2)
        a_expanded = encoded_a.unsqueeze(2) # (b, len_a,1, 2h)
        a_expanded = a_expanded.expand(shape) # (b, len_a, len_c, 2h)
        c_expanded = encoded_c.unsqueeze(1) # (b,1, len_c, 2h)
        c_expanded = c_expanded.expand(shape) # (b, len_a, len_c, 2h)

        a_elmwise_mul_c = torch.mul(a_expanded, c_expanded) # (b, len_a, len_c, 2h)
        cat_data = torch.cat((a_expanded, c_expanded, a_elmwise_mul_c), 3) # (b, len_a,len_c, 6h), [h;u;h◦u]
        S = self.Wac(cat_data).view(batch_size, len_a, len_c) # (b, len_a, len_c)

        # a2b
        a2c = torch.bmm(F.softmax(S, dim=-1), encoded_c) # (b, len_a, 2h) = bmm( (b, len_a,len_c), (b, len_c, 2h) )
        # c2a
        # c: attention weights on the context
        c = F.softmax(torch.max(S, 2)[0], dim=-1) # (b, len_c)
        c2a = torch.bmm(c.unsqueeze(1), encoded_a) # (b, 1, 2d) = bmm( (b, 1, len_c), (b, len_c, 2d) )
        c2a = c2a.repeat(1, len_c, 1) # (b, len_c, 2d), tiled len_c times

       #finally b2c and c2b
        shape = (batch_size,len_b,len_c,h2)
        b_expanded = encoded_b.unsqueeze(2) # (b, len_a,1, 2h)
        b_expanded = b_expanded.expand(shape) # (b, len_a, len_c, 2h)
        c_expanded = encoded_c.unsqueeze(1) # (b,1, len_c, 2h)
        c_expanded = c_expanded.expand(shape) # (b, len_a, len_c, 2h)

        b_elmwise_mul_c = torch.mul(b_expanded, c_expanded) # (b, len_a, len_c, 2h)
        cat_data = torch.cat((b_expanded, c_expanded, b_elmwise_mul_c), 3) # (b, len_a,len_c, 6h), [h;u;h◦u]
        S = self.Wbc(cat_data).view(batch_size, len_b, len_c) # (b, len_a, len_c)

        # b2c
        b2c = torch.bmm(F.softmax(S, dim=-1), encoded_c) # (b, len_a, 2h) = bmm( (b, len_a,len_c), (b, len_c, 2h) )
        # c2b
        # c: attention weights on the context
        c = F.softmax(torch.max(S, 2)[0], dim=-1) # (b, len_b)
        c2b = torch.bmm(c.unsqueeze(1), encoded_b) # (b, 1, 2d) = bmm( (b, 1, len_c), (b, len_b, 2d) )
        c2b = c2b.repeat(1, len_c, 1) # (b, len_c, 2d), tiled len_c times

        att_enc_a = torch.cat((encoded_a, a2b, a2c, encoded_a.mul(a2b), encoded_a.mul(a2c)),2)
        att_enc_b = torch.cat((encoded_b, b2a, b2c, encoded_b.mul(b2a), encoded_b.mul(b2c)),2)
        att_enc_c = torch.cat((encoded_c, c2a, c2b, encoded_c.mul(c2a), encoded_c.mul(c2b)),2) 

        all_att = torch.cat((att_enc_a,att_enc_b,att_enc_c),1) # b, len_a + len_b + len_c, 10d
        return all_att



    def encode_part(self, source:List[List[str]],encode_input: str):

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
 
        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)   # Tensor: (src_len, b)
        char_embs = self.model_char_embeddings_source(source_padded_chars)
        word_embs = self.model_word_embeddings.source(source_padded)

        embeds = self.contextual_embedding_layer(word_embs,char_embs) #(src_len,b, self.d)


        source_lengths = [len(s) for s in source]
        source_lengths = torch.tensor(source_lengths)


        orig_len = embeds.size(0)
        # embed_mask = torch.zeros_like(embeds) != embeds
        # source_lengths = embed_mask.sum(-1)

        source_lengths, sort_idx = source_lengths.sort(0, descending=True)
        embeds = embeds[:,sort_idx,:]   
        # Unpack and reverse sort
        pack_padded_X = pack_padded_sequence(embeds, source_lengths) 
    
        if encode_input == 'A':
            packed_enc_hiddens, (last_hidden, last_cell) = self.encoder_a(pack_padded_X)
            enc_hiddens, _ = pad_packed_sequence(packed_enc_hiddens,total_length=orig_len)
            enc_hiddens = enc_hiddens.permute(1, 0, 2)
            _, unsort_idx = sort_idx.sort(0)
            enc_hiddens = enc_hiddens[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

            init_decoder_hidden = self.ha_projection(torch.cat((last_hidden[0], last_hidden[1]), 1))
            init_decoder_cell = self.ca_projection(torch.cat((last_cell[0], last_cell[1]), 1))
        elif encode_input == 'B':
            packed_enc_hiddens, (last_hidden, last_cell) = self.encoder_b(pack_padded_X)
            enc_hiddens, _ = pad_packed_sequence(packed_enc_hiddens,total_length=orig_len)
            enc_hiddens = enc_hiddens.permute(1, 0, 2)
            _, unsort_idx = sort_idx.sort(0)
            enc_hiddens = enc_hiddens[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

            init_decoder_hidden = self.hb_projection(torch.cat((last_hidden[0], last_hidden[1]), 1))
            init_decoder_cell = self.cb_projection(torch.cat((last_cell[0], last_cell[1]), 1))
        elif encode_input == 'C':
            packed_enc_hiddens, (last_hidden, last_cell) = self.encoder_c(pack_padded_X)
            enc_hiddens, _ = pad_packed_sequence(packed_enc_hiddens,total_length=orig_len)
            enc_hiddens = enc_hiddens.permute(1, 0, 2)
            _, unsort_idx = sort_idx.sort(0)
            enc_hiddens = enc_hiddens[unsort_idx]  # (batch_size, seq_len, 2 * hidden_size)

            init_decoder_hidden = self.hc_projection(torch.cat((last_hidden[0], last_hidden[1]), 1))
            init_decoder_cell = self.cc_projection(torch.cat((last_cell[0], last_cell[1]), 1))
        else:
            Raise('Error- need an encoding')


        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        ### END YOUR CODE

        return enc_hiddens, dec_init_state



    def forward_o(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.src.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)   # Tensor: (src_len, b)
        target_padded_chars = self.vocab.src.to_input_tensor_char(target, device=self.device)   # Tensor: (tgt_len, b)

        char_embs = self.model_char_embeddings_source(source_padded_chars)
        word_embs = self.model_word_embeddings.source(source_padded)

        embed = self.contextual_embedding_layer(word_embs,char_embs)
        # print(embed.shape)
        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(embed, source_lengths)
        # print(enc_hiddens.shape)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        # logits =self.target_vocab_projection(combined_outputs)

        # loss = nn.CrossEntropyLoss(reduction='sum',ignore_index=self.vocab.tgt['<pad>'])
        # xentropy_loss = loss(logits.permute(0,2,1),target_padded[1:])
        # return xentropy_loss
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.src['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum() # mhahn2 Small modification from A4 code.
    
        return scores


    def forward(self, A: List[List[str]],B: List[List[str]],C: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        target_padded = self.vocab.src.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)
        target_padded_chars = self.vocab.src.to_input_tensor_char(target, device=self.device)   # Tensor: (tgt_len, b)


        # print(embed.shape)
        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        source_lengths = [[len(s) for s in A], [len(s) for s in B], [len(s) for s in C]]

        encoded_a, dec_init_state_a = self.encode_part(A,'A')
        encoded_b, dec_init_state_b = self.encode_part(B,'B')
        encoded_c, dec_init_state_c = self.encode_part(C,'C')


        dec_init_state_h = torch.cat((dec_init_state_a[0], dec_init_state_b[0], dec_init_state_c[0]),1)
        dec_init_state_c = torch.cat((dec_init_state_a[1], dec_init_state_b[1], dec_init_state_c[1]),1)



        # dec_init_state = (dec_init_state_h,dec_init_state_c)

        # dec_init_state_h = dec_init_state_a[0] + dec_init_state_b[0] + dec_init_state_c[0]
        # dec_init_state_c = dec_init_state_a[0] + dec_init_state_b[0] + dec_init_state_c[0]

        dec_init_state = (dec_init_state_h,dec_init_state_c)

        #change up dec_init_state = (init_decoder_hidden, init_decoder_cell)

        enc_hiddens = self.tri_dir_att(encoded_a,encoded_b,encoded_c)

        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.src['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum() # mhahn2 Small modification from A4 code.
    
        return scores

    def contextual_embedding_layer(self,p_word_source: torch.Tensor, p_char_source:torch.Tensor) -> torch.Tensor:
        #p_word_source
        # Highway Networks for 1. and 2.
        embd = torch.cat((p_word_source, p_char_source), -1) # (batch, max_sent_len, embd_size*2)
        embd = self.highway(embd) # (batch, max_sent_len, embd_size*2)

        return embd


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        ###
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        Y = self.model_word_embeddings.source(target_padded) #cause source_padded already a tensor. #major error
        for Y_t in torch.split(Y,1):
            Y_t = torch.squeeze(Y_t)
            # post concat it should be (b,h+e)
            Ybar_t = torch.cat((Y_t, o_prev), 1)
            dec_state, o_t, _ = self.step(Ybar_t,dec_state,enc_hiddens,enc_hiddens_proj,enc_masks)
            combined_outputs.append(o_t)
            o_prev = o_t
        combined_outputs = torch.stack(combined_outputs,0)
        ### END YOUR CODE

        return combined_outputs


    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze

        dec_state = self.decoder(Ybar_t, dec_state)
        (dec_hidden, dec_cell) = dec_state  
        e_t = torch.squeeze(torch.bmm(enc_hiddens_proj, torch.unsqueeze(dec_hidden,2)),2)
        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh

        alpha_t = nn.functional.softmax(e_t,1)
        a_t = torch.squeeze(torch.bmm(torch.unsqueeze(alpha_t,1), enc_hiddens),1)
        U_t = torch.cat((dec_hidden,a_t),1)
        V_t = self.combined_output_projection(U_t)
        O_t = self.dropout(torch.tanh(V_t))
        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[List[int]]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)

        for e_id, src_len in enumerate(source_lengths):
            A_max = len(source_lengths[0])
            B_max = len(source_lengths[1])
            C_max = len(source_lengths[2])

            A_src_len = source_lengths[0][e_id]
            B_src_len = source_lengths[1][e_id]
            C_src_len = source_lengths[2][e_id]

            enc_masks[e_id, A_src_len:A_max] = 1
            enc_masks[e_id, B_src_len:B_max] = 1
            enc_masks[e_id, C_src_len:C_max] = 1
        return enc_masks.to(self.device)


    def beam_search(self, A: List[str], B: List[str], C: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        # Convert list of lists into tensors
        encoded_a, dec_init_state_a = self.encode_part([A],'A')
        encoded_b, dec_init_state_b = self.encode_part([B],'B')
        encoded_c, dec_init_state_c = self.encode_part([C],'C')


        dec_init_state_h = torch.cat((dec_init_state_a[0], dec_init_state_b[0], dec_init_state_c[0]),1)
        dec_init_state_c = torch.cat((dec_init_state_a[1], dec_init_state_b[1], dec_init_state_c[1]),1)




        dec_init_state = (dec_init_state_h,dec_init_state_c)
        

        #change up dec_init_state = (init_decoder_hidden, init_decoder_cell)

        src_encodings_att_linear = self.tri_dir_att(encoded_a,encoded_b,encoded_c)
        src_encodings_att_linear = self.att_projection(src_encodings)
    




        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.src['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            # y_tm1 = torch.tensor([self.vocab.src[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            # y_t_embed = self.model_word_embeddings.source(y_tm1)

            y_tm1 = self.vocab.src.to_input_tensor_char(list([hyp[-1]] for hyp in hypotheses), device=self.device)
            y_t_embed = self.model_word_embeddings.source(y_tm1)
            y_t_embed = torch.squeeze(y_t_embed, dim=0)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.src)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.src)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.src.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_word_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str,no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'],no_char_decoder=False, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_word_embeddings.embed_size, hidden_size=self.hidden_size, weights=self.model_word_embeddings.source.weight, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
