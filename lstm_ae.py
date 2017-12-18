from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.core import RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import collections
import nltk
import numpy as np
import os
import lexicon
from nltk.corpus import stopwords
import history_plot
from sklearn.model_selection import train_test_split
import math

# read sentences from file, tabular sentences in sent_filename

# DATA_DIR = 'Data/'
# sent_filename = os.path.join(DATA_DIR,'data.txt')

DATA_DIR = 'D:/Dataset/amazon-5core/'
sent_filename = os.path.join(DATA_DIR,'amazon_train.txt')


# stopwords
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}'])

sents = []
fsent = open(sent_filename, "r")
for line in fsent:
    sent = line.strip()
    sents.append(sent)
fsent.close()

index = np.arange(len(sents))
np.random.shuffle(index)
new_sents = []
for i in index.tolist():
    new_sents.append(sents[i])

sents = new_sents

def is_number(n):
    temp = nltk.re.sub("[.,-/]", "", n)
    return temp.isdigit()


# pre-process sentences, replace digits with 9, lowercase, prepare sentence lengths

word_freqs = collections.Counter()
sent_lens = []
parsed_sentences = []
for sent in sents:
    words = nltk.word_tokenize(sent)
    parsed_words = []
    for word in words:
        if ~is_number(word) and word not in stop_words:
            word_freqs[word.lower()] += 1
            parsed_words.append(word.lower())
    sent_lens.append(len(parsed_words))
    parsed_sentences.append(" ".join(parsed_words))

# export corpus statistics

sent_lens = np.array(sent_lens)
print("number of sentences: {:d}".format(len(sent_lens)))
print("distribution of sentence lengths (number of words)")
print("min:{:d}, max:{:d}, mean:{:.3f}, med:{:.3f}".format(np.min(sent_lens), np.max(sent_lens),
                                                           np.mean(sent_lens), np.median(sent_lens)))
print("vocab size (full): {:d}".format(len(word_freqs)))

# LSTM hyper parameters, vocab size is set to cover 90% of all the corpus
# sequence length is set to twice of median, 90% of sentences are less than this value
# other words will be assigned UNK, sentences with larger sequence size will be assigned a special seq pad

VOCAB_SIZE = 100000
SEQUENCE_LEN = 60

# VOCAB_SIZE = 5000
# SEQUENCE_LEN = 30



# the input to LSTM is numeric, so we use a lookup table
# lookup table: word2id and id2word

word2id = {}
word2id["PAD"] = 0
word2id["UNK"] = 1
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v:k for k, v in word2id.items()}

# in order to present each word we use Glove word embeddings instead of one-hot-vector, because 1-hot will make it large
# Glove is 50 dim, DATA_DIR: glove folder

EMBED_SIZE = 50



def lookup_word2id(word):
    try:
        return word2id[word]
    except KeyError:
        return word2id["UNK"]


def load_glove_vectors(glove_file, word2id, embed_size):
    embedding = np.zeros((len(word2id), embed_size))
    fglove = open(glove_file, "r", encoding="utf8")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        if embed_size == 0:
            embed_size = len(cols) - 1
        if word in word2id:
            vec = np.array([float(v) for v in cols[1:]])
        embedding[lookup_word2id(word)] = vec
    embedding[word2id["PAD"]] = np.zeros((embed_size))
    embedding[word2id["UNK"]] = np.random.uniform(-1, 1, embed_size)
    return embedding

embeddings = load_glove_vectors(os.path.join(
    'Data/', "glove.6B.{:d}d.txt".format(EMBED_SIZE)), word2id, EMBED_SIZE)

# generator will shuffle sentences at the beginning of each epoch, returns 64 batch sentences
# each sentence is represented with Glove vectors
# two generators are defined, one for train and one for test (70% and 30%)

BATCH_SIZE = 512

sent_wids = np.zeros((len(parsed_sentences),SEQUENCE_LEN),'int32')
sample_seq_weights = np.zeros((len(parsed_sentences),SEQUENCE_LEN),'float')
for index_sentence in range(len(parsed_sentences)):
    temp_sentence = parsed_sentences[index_sentence]
    temp_words = nltk.word_tokenize(temp_sentence)
    for index_word in range(SEQUENCE_LEN):
        if index_word < sent_lens[index_sentence]:
            sent_wids[index_sentence,index_word] = lookup_word2id(temp_words[index_word])
            sample_seq_weights[index_sentence,index_word] = lexicon.get_word_sentiment_weight(temp_words[index_word])
        else:
            sent_wids[index_sentence, index_word] = lookup_word2id('PAD')

def sentence_generator(X, embeddings, batch_size, sample_weights):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size : (bid + 1) * batch_size]
            temp_sents = X[sids, :]
            Xbatch = embeddings[temp_sents]
            weights = sample_weights[sids, :]
            yield Xbatch, Xbatch


train_size = 0.95
split_index = int(math.ceil(len(sent_wids)*train_size))
# split = np.array(split_index,dtype='int32')
# X = np.split(sent_wids,split)
# weights = np.split(sample_seq_weights,split)
Xtrain = sent_wids[0:split_index,:]
Xtest = sent_wids[split_index:,:]
train_w = sample_seq_weights[0:split_index,:]
test_w = sample_seq_weights[split_index:,:]
# Xtrain, Xtest = train_test_split(sent_wids, train_size=train_size)
train_gen = sentence_generator(Xtrain, embeddings, BATCH_SIZE,train_w)
test_gen = sentence_generator(Xtest, embeddings, BATCH_SIZE,test_w)


# LSTM model construction
# repeat layer will replicate output of encoder to generate (batch_size, sequence_length, embedding_size) for decoder


LATENT_SIZE = 100
# LATENT_SIZE = 10

inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", name="encoder_lstm")(inputs)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True), merge_mode="sum", name="decoder_lstm")(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="sgd", loss='mse')
# autoencoder.compile(optimizer="sgd", loss='mse', sample_weight_mode='temporal')
# autoencoder.compile(optimizer="sgd", loss='mse')
# train the model for 10 epochs and save the best model based on MSE loss

NUM_EPOCHS = 5

num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE
checkpoint = ModelCheckpoint(filepath=os.path.join('Data/', "simple_ae_to_compare"), save_best_only=True)
history = autoencoder.fit_generator(train_gen, steps_per_epoch=num_train_steps, epochs=NUM_EPOCHS,
                                    validation_data=test_gen, validation_steps=num_test_steps, callbacks=[checkpoint])
history_plot.plot(history)


