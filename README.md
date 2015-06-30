# About
+ **char_rnn_theano_orig.py** is code for replicating the model they had with a few differences we can discuss.
+ **char_rnn_theano.py** is the same, but with random parameter seeding to search over for the best network and training configuration.  
+ **gen_patternltd_data.py** is a quick and dirty generator of a toy dataset that I think stands in well for the problem of predicting target outcomes that require a good memory.  In this case, the dataset is a string of digits.  The network has to learn the rule that, if it sees ‘456’ within 20 steps of seeing ‘123’, then it has to switch it’s output from 0 to 1.  I also added predictors of ‘123’ and ‘456’ (‘7’, and ‘8’, respectively), that have a relatively strong probability (0.5) of being followed immediately by the patterns.  I haven’t started running this yet, however, so I don’t have the runner script ready for you.
+ input.txt is the shakespear data
+ jbpickle.py is a pickling helper
+ submit* are wrappers for submitting to princeton gpu clusters


# TODO
+ merge char_rnn_theano_* and add flags for differences instead
+ add model comparison script
+ add clipping of gradients to keras opimizer
+ modularize data prep out
+ make runner for patternltd dataset
+ create early stopping crit for bad training
