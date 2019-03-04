from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pprint

glove_file = './data/word_vectors/life_vectors.txt'
tmp_file = get_tmpfile("word2vec_life.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)
# cellulose | Adhesion | plant | Photosynthesis 
pprint.pprint(model.most_similar(positive=['adhesion', 'plant'], negative=['cellulose']))