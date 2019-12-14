import numpy as np
from gensim.models.keyedvectors import KeyedVectors

word2vec_path_bin = '/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/Medical Data/embeddings/bio_embedding_extrinsic.bin'
word2vec_path_txt = '/media/ramkabir/PC Data/ASU Data/Semester 3/BMNLP/Projects/Medical Data/embeddings/bio_embedding_extrinsic.txt'

counter = 0
limit = 10

# model = KeyedVectors.load_word2vec_format(word2vec_path_bin, binary=True)
# model.save_word2vec_format(word2vec_path_txt, binary=False)

with open(word2vec_path_txt) as f:
    for line in f:
        if counter == limit:
            break
        else:
            print(line)
            counter += 1