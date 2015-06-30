import numpy as np
import numpy.random as rng
from numpy import array as npa
import re




def find_all(pat, s):
    return [m.start() for m in re.finditer(pat, s)]



# create training set
# for iseq in xrange(NSEQ):

def create_y(sstream, istart, seqlen, spat1, spat2, delay_at_most):
    y = np.zeros(seqlen)
    # istart = rng.randint(len(stream-seqlen))
    ifinish = istart + seqlen
    seq = sstream[istart:ifinish]
    s1at = find_all(spat1, seq)
    for is1 in s1at:
        iend = min(is1+delay_at_most, seqlen)
        subseq = seq[is1:iend]
        s2at = subseq.find(spat2)
        if s2at > -1:
            y[is1+s2at+len(spat2):] = 1
    return y

        
def onehot(i, n):
    oh = np.zeros(n)
    oh[i] = 1.
    return oh

def create_x(seq_string):
    return np.vstack([onehot(int(i), 10) for i in seq_string])


# create
def toy_dataset():
    PATTERN1 = npa([1,2,3])
    DELAY_AT_MOST = 20  # trouble if pattern2 occurs within this many steps after pattern 1
    PATTERN2 = npa([4,5,6])
    maxpatlen = max(len(PATTERN1), len(PATTERN2))

    PRED_PAT1 = 7  # predicts pattern 1
    P_PRED_PAT1 = 0.5
    PRED_PAT2 = 8  # predicts pattern 2
    P_PRED_PAT2 = 0.5

    SEQLEN = 500
    NSEQ = 10000

    stream = rng.randint(10, size=1000000)
    # add prediction
    for i, v in enumerate(stream[:-maxpatlen]):
        if v == PRED_PAT1:
            if rng.rand() < P_PRED_PAT1:
                stream[i+1:i+1+len(PATTERN1)] = PATTERN1
        if v == PRED_PAT2:
            if rng.rand() < P_PRED_PAT2:
                stream[i+1:i+1+len(PATTERN2)] = PATTERN2
        istarts = rng.randint(len(stream)-SEQLEN, size=NSEQ)

    # find stringify
    sstream = ''.join([str(i) for i in stream.tolist()])
    spat1 = ''.join([str(i) for i in PATTERN1.tolist()])
    spat2 = ''.join([str(i) for i in PATTERN2.tolist()])

    start_pat1 = find_all(spat1, sstream)
    start_pat2 = find_all(spat2, sstream)
    y = np.vstack([create_y(sstream, istart, SEQLEN, spat1, spat2, DELAY_AT_MOST) \
                for istart in istarts])
    X = [create_x(sstream[istart:istart+SEQLEN]) for istart in istarts]
    X = np.dstack(X)
    X = np.rollaxis(X, -1)  # now xcase x seqlen x 10

    return X, y


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
