import cPickle
def pickle(obj, fname):
    """obj is the session object you're saving.  fname is the file name"""
    with open(fname, 'wb') as f:
        cPickle.dump(obj, f)

def unpickle(fname):
    """fname is the file name.  you have to assign to an object.
    e.g. unpickle('fname.py'), where file was saved as pickle(obj, 'fname.py')
    will not put the object obj in your environment.  you need to do
    obj = unpickle('fname.py')"""
    with open(fname) as f:
        obj = cPickle.load(f)

    return obj

