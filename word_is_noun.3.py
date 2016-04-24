# -*- coding: utf-8 -*-
'''An implementation of sequence to sequence learning for performing addition
Input: "535+61"
Output: "596"
Padding is handled by using a repeated sentinel character (space)
Input may optionally be inverted, shown to increase performance in many tasks in:
"Learning to Execute"
http://arxiv.org/abs/1410.4615
and
"Sequence to Sequence Learning with Neural Networks"
http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
Theoretically it introduces shorter term dependencies between source and target.
Two digits inverted:
+ One layer LSTM (128 HN), 5k training examples = 99% train/test accuracy in 55 epochs
Three digits inverted:
+ One layer LSTM (128 HN), 50k training examples = 99% train/test accuracy in 100 epochs
Four digits inverted:
+ One layer LSTM (128 HN), 400k training examples = 99% train/test accuracy in 20 epochs
Five digits inverted:
+ One layer LSTM (128 HN), 550k training examples = 99% train/test accuracy in 30 epochs
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers import recurrent
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.callbacks
import numpy as np
from six.moves import range
import sys



class HistoryCallback(keras.callbacks.Callback):

    # ужасная реализация для вычисления точности для бинарной классификации
    # добавить вычисление F1
    def on_epoch_end(self, batch, logs={}):
     y_pred = model.predict( X_test, verbose=0 )
     n_error=0
     n_success=0
     for i in range(y_pred.shape[0]):
         if y_pred[i,0]>y_pred[i,1]:
             if y_test[i,0]==1:
                 n_success = n_success+1
             else:
                 n_error = n_error+1
         else:
             if y_test[i,1]==1:
                 n_success = n_success+1
             else:
                 n_error = n_error+1

     print( 'err=', float(n_error)*100.0/float(n_error+n_success), '%' )



class CharacterTable(object):
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilties to their character output
    '''
    def __init__(self, chars, maxlen):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.maxlen = maxlen

    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

# Parameters for the model and dataset
TRAINING_SIZE = 200000
INVERT = True
# Try replacing GRU, or SimpleRNN
RNN = recurrent.LSTM
HIDDEN_SIZE = 512
BATCH_SIZE = 128
LAYERS = 1

corpus_path = 'word_is_noun.dat'

print('Loading data', corpus_path, '...')

patterns = []
max_word_len = 0

chars = set([u' '])
with open( corpus_path, 'r' ) as fdata:
    
    fdata.readline()
    
    while len(patterns) < TRAINING_SIZE:
        
        toks = fdata.readline().strip().decode('utf-8').split('\t')
        
        word = toks[0]
        is_noun = int(toks[2])
        
        if ' ' not in word:
            max_word_len = max( max_word_len, len(word) )
            patterns.append( (word,is_noun) )
            chars.update( list(word) )

ctable = CharacterTable(chars, max_word_len)
        
print('Total number of patterns:', len(patterns))
print('max_word_len=', max_word_len );

questions = []
expected = []

for ipattern,pattern in enumerate(patterns):
    
    # Pad the data with spaces such that it is always MAXLEN
    q = pattern[0]
    query = q + ' ' * (max_word_len - len(q))
    if INVERT:
        query = query[::-1]
    
    answer = pattern[1]
    
    questions.append(query)
    expected.append(answer)

n_patterns = len(questions)
test_share = 0.1
n_test = int(n_patterns*test_share)
n_train = n_patterns-n_test

print('Vectorization...')
X_train = np.zeros((n_train, max_word_len, len(chars)), dtype=np.bool)
y_train = np.zeros((n_train, 2), dtype=np.bool)

X_test = np.zeros((n_test, max_word_len, len(chars)), dtype=np.bool)
y_test = np.zeros((n_test, 2), dtype=np.bool)

i_test = 0
i_train = 0
for i in range(len(questions)):

    word = questions[i]
    is_noun = expected[i]

    if i<n_test:
        X_test[i_test] = ctable.encode(word, maxlen=max_word_len)
        y_test[i_test,is_noun] = 1
        i_test = i_test+1
    else:
        X_train[i_train] = ctable.encode(word, maxlen=max_word_len)
        y_train[i_train,is_noun] = 1
        i_train = i_train+1
        
print(X_train.shape)
print(y_train.shape)


print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(max_word_len, len(chars))))

model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

hist = HistoryCallback()
model_checkpoint = ModelCheckpoint( 'word_is_noun.model', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping( monitor='val_loss', patience=10, verbose=1, mode='auto')

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=30, validation_data=(X_test, y_test), callbacks=[model_checkpoint,early_stopping,hist])
with open( 'performance.txt', 'a' ) as f:
    f.write( str(history)+'\n' )
