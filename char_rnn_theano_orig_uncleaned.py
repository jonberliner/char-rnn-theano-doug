from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from copy import deepcopy
from numpy import array as npa
import numpy as np
from numpy.random import RandomState
from theano import tensor as T
import string
import sys
from jbpickle import pickle, unpickle
import os

SAVEDIR = './out/'


rng = RandomState()  # will reseed after param selection

# random number string for book keeping
pool = string.ascii_uppercase + string.digits
RUNID = 'ORIGP_' + ''.join([str(pool[rng.randint(len(pool))]) for _ in xrange(8)])

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

SAVEDIR = SAVEDIR + RUNID + '/'
WDIR = SAVEDIR + 'W/'
mkdir(SAVEDIR)
mkdir(WDIR)
PRINTTO = SAVEDIR + 'generated_output.txt'


SEQLEN = 50
SHUF_DATA = 0
NLAYER = 2
NHID = [128, 128]
PDROP = [0., 0.]
BATCHSIZE = 50
LRINIT = 2e-3
RNGSEED = rng.randint(4525348)

rng = RandomState(RNGSEED)

runparams = {'SEQLEN': SEQLEN,
             'SHUF_DATA': SHUF_DATA,
             'NLAYER': NLAYER,
             'NHID': NHID,
             'PDROP': PDROP,
             'BATCHSIZE': BATCHSIZE,
             'LRINIT': LRINIT,
             'RNGSEED': RNGSEED}

pickle(runparams, SAVEDIR+'runparams.pkl')

print 'RUNID: ' + RUNID
for k, v in runparams.items():
    print k + ': ' + str(v)


# save params for book keeping
with open(PRINTTO, 'w+') as f:
    for k,v in runparams.items():
        f.write('\n')
        f.write('%s: %s' % (k,v))
        f.write('\n\n')

NB_EPOCH = 1
NEPOCH = 50
LRDECAY = 0.97
LRDECAYAFTER = 10

FNAME = 'input.txt'
PTRAIN = 0.7
PVAL = 0.15
PTEST = 0.15

# x will be text.  y will be same thing shifted by 1
def load_text(fname):
    lochars = []
    chars = {}
    ichar = -1
    with open(fname) as f:
        while True:
            c = f.read(1)
            if not c: break  # exit if file over
            if c not in chars:
                ichar += 1
                chars[c] = ichar
            lochars.append(c)
    return (lochars, chars)


def lochars_to_mats(text, char2i):
    ntot = len(text)
    nchar = len(char2i)
    X = np.zeros([ntot, nchar])
    for ic,c in enumerate(text):
        X[ic, char2i[c]] = 1.
    y = deepcopy(X)
    y = np.vstack([y[-1], y[:-1]])  # wraps around to beginning, i guess
    return (X, y)


def seqify(X, y, seqlen):
    ntot, nchar = X.shape
    nseq = ntot / seqlen
    clip =  ntot % seqlen  # remove dangling text
    if clip != 0:
        X = X[:-clip]  # make so can have even seqlens
        y = y[:-clip]  # make so can have even seqlens
    Xseq = np.zeros([nseq, seqlen, nchar])
    # yseq = np.zeros([nseq, seqlen, nchar])
    yseq = np.zeros([nseq, nchar])
    for iseq in xrange(nseq):
        Xseq[iseq] = X[seqlen*iseq:seqlen*(iseq+1)]
        if iseq==nseq-1: yseq[iseq] = X[0]
        else: yseq[iseq] = X[seqlen*(iseq+1)]
        # yseq[iseq] = y[seqlen*iseq:seqlen*(iseq+1)]
    return Xseq, yseq


def trainvaltest_split(Xseq, yseq, p_train, p_val, p_test=None, shuffle=False):
    ncase, seqlen, nbatch = Xseq.shape
    if not p_test: p_test = 1.-(p_train+p_val)
    # can be less if don't want to use full dataset
    assert npa([p_train, p_val, p_test]).sum() <= 1.
    ntrain, nval, ntest = [int(p * ncase) for p in [p_train, p_val, p_test]]

    if shuffle:
        neworder = rng.permutation(ncase)
        Xseq = Xseq[neworder]
        yseq = yseq[neworder]

    return {'train': {'X': Xseq[:ntrain], 'y': yseq[:ntrain]},
            'val': {'X': Xseq[ntrain:ntrain+nval], 'y': yseq[ntrain:ntrain+nval]},
            'test': {'X': Xseq[ntrain+nval:ntrain+nval+ntest], 'y': yseq[ntrain+nval:ntrain+nval+ntest]}
           }


# for generation
def onehot_to_char(onehot, i2char):
    return i2char[onehot.argmax()]

def char_to_onehot(char, char2i):
    onehot = np.zeros(len(char2i))
    onehot[char2i[char]] = 1.
    return onehot


def softmax(arr, temp):
    assert temp >= 0
    if temp == 0:
        out = np.zeros_like(arr)
        out[arr.argmax()] = 1.
    else:
        out = np.exp(arr/temp)
        out /= out.sum()
    return out


def sample(arr, temp):
    return rng.choice(np.arange(len(arr)), p=softmax(arr, temp))




print 'prepping data'
lochars, char2i = load_text(FNAME)
i2char = {v: k for k, v in char2i.items()}
nchar = len(char2i)
X, y = lochars_to_mats(lochars, char2i)
Xseq, yseq = seqify(X, y, SEQLEN)
tvt = trainvaltest_split(Xseq, yseq, PTRAIN, PVAL, PTEST, SHUF_DATA)
if not NB_EPOCH:
    NB_EPOCH = tvt['train']['X'].shape[0] / BATCHSIZE


# DEFINE MODEL
model = Sequential()
# add stacked lstm layers
for ilayer in xrange(NLAYER):
    if ilayer==0: NIN = nchar
    else: NIN = NHID[ilayer-1]
    RET_SEQ = ilayer != NLAYER-1
    model.add(LSTM(NIN, NHID[ilayer],\
                   return_sequences=RET_SEQ,\
                   activation='tanh',\
                   inner_activation='hard_sigmoid'))
    if PDROP[ilayer] != 0:
        model.add(Dropout(PDROP[ilayer]))
# readout layer
model.add(Dense(NHID[-1], nchar))
# def logsoftmax(x):
#     return T.log(T.nnet.softmax(x))
model.add(Activation('softmax'))

print 'compling model...'
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

for ie in xrange(NEPOCH):
    print ''.join(['epoch ', str(ie)])
    if ie < LRDECAYAFTER:
        model.optimizer.lr = LRINIT
    else:
        model.optimizer.lr *= LRDECAY

    trainscores = model.fit(tvt['train']['X'], tvt['train']['y'],\
                            batch_size=BATCHSIZE,
                            nb_epoch=NB_EPOCH)
    wname = ''.join([WDIR, 'epoch_', str(ie), '.hdf5'])
    model.save_weights(wname)
    valscore = model.evaluate(tvt['val']['X'], tvt['val']['y'],\
                              batch_size=BATCHSIZE)

    # print out for human eval
    outlen = 600  # num char to predict out
    istart = rng.randint(tvt['test']['X'].shape[0]-outlen-1)
    for temp in np.linspace(0., 4., 9):
        print()
        print('----- temp:', temp)
        print '----- epoch:' + str(ie)
        starter = X[istart:istart+SEQLEN]
        generated = ''.join([onehot_to_char(oh, i2char) for oh in starter])
        inittxt = deepcopy(generated)

        print('----- Generating with seed: "' + generated + '"')
        sys.stdout.write(generated)

        for ig in xrange(outlen):
            preds = model.predict(starter[None,:], verbose=0)[0]
            next_ichar = sample(preds, temp)
            next_char = i2char[next_ichar]
            generated += next_char
            starter = np.vstack([starter[1:], char_to_onehot(next_char, char2i)])

            # print to cmd
            sys.stdout.write(next_char)
            sys.stdout.flush()

        with open(PRINTTO, 'a') as f:
            f.write('\ngenerated with seed \'%s\n\'' % inittxt)
            f.write('\n\nepoch %d, temp=%d:\n\n' % (ie, temp))
            f.write(generated)



