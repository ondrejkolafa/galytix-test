import gensim
from gensim.models import KeyedVectors


LOCATION = "resources\GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_PATH = "resources/word2vec.csv"

wv = KeyedVectors.load_word2vec_format(LOCATION, binary=True, limit=1000000)
wv.save_word2vec_format(WORD2VEC_PATH)