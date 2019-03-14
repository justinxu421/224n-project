import torch
# need specials and stuff
def w2v_embeding(wordvecs,train_word2index):
    sents = []

    print("Reading in pretrained word vectors")
    weight_embeddings = torch.zeros(4,50)
    index = 4

    word2index = {}
    word2index['<pad>'] = 0
    word2index['<unk>'] = 1
    word2index['<s>'] = 2
    word2index['</s>'] = 3


    index2word = {}
    index2word[0] = '<pad>'
    index2word[1] = '<unk>'
    index2word[2] = '<s>'
    index2word[3] = '</s>'

    with open(wordvecs) as f:
        for sent in f:
            cur_sent = sent[:]
            cur_sent = cur_sent.lower()
            cur_sent = sent.strip().split(' ')

            word = cur_sent[0]
            if word in word2index: continue;
            index2word[index] = word
            word2index[word] = index

            weight_dim = len(cur_sent[1:])
            weights = [[float(weight) for weight in cur_sent[1:]]]
            torch_w = torch.tensor(weights)
            weight_embeddings = torch.cat((weight_embeddings,torch_w),0)
            # print(weight_embeddings.shape)
            index += 1


    additional_words  = [ word for word in train_word2index if word not in word2index]
    for word in additional_words:
        torch_w = torch.zeros(1,50)
        weight_embeddings = torch.cat((weight_embeddings,torch_w),0)      
        index2word[index] = word
        word2index[word] = index
        index += 1
    print(weight_embeddings.shape)
    return weight_embeddings,word2index,index2word

# print('hi')
# weight_embeddings,word2index,index2word = w2v_embeding('../GloVe-1.2/life_vectors.txt')
# print(weight_embeddings.shape)
# print(len(word2index))

