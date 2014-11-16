"""
"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from mlp import HiddenLayer
from jdy_A import A

from jdy_utils import load_data, save_short, save_med_pkl, save_med_npy



class DAfinetune(object):
    """Deep Autoencoder (easiest to think of as an unsupervised deep neural 
    network) with unsupervised global backpropagation for fine-tuning an already
    pre-trained model (can also run without pretraining).

    This is the fine-tuning (DAfinetune) step of a Srbm_DAfinetune. jdy_SA.py is a stacked 
    autoencoder that is trained layber-by-layer and should only be used for 
    pretraining.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500]):
        ### removed 'corruption_levels=[0.1, 0.1]' from def __init__
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the SA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.hidden_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        
        #create list of hidden layer output sizes
        decoder_layers = hidden_layers_sizes[:]
        decoder_layers.reverse()
        unrolled_hidden_layers_sizes = hidden_layers_sizes + decoder_layers[1:]


        for i in xrange(len(unrolled_hidden_layers_sizes)):
            # construct the hidden layers

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = unrolled_hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the SRBM_SA if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.hidden_layers[-1].output  
                ### could change to self.rbm_layers[-1].propup. then i wouldn't 
                ### have to create self.hidden_layers

            ### TODO:need to initialize each of these layers with the parameters 
            ### learned during pretraining. When get these weights should pass 
            ### them as values (.get_value) not as symbolic variables
            hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=unrolled_hidden_layers_sizes[i], 
                                        activation=T.nnet.sigmoid)
            ###could create a file of user-defined activation functions. 
            ###activation function could be passed to DAfinetune init

            # add the layer to our list of layers
            self.hidden_layers.append(hidden_layer) 

            # add weights and biases to self.params
            self.params.extend(hidden_layer.params) 

        # create reconstruction layer
        self.reconstructionLayer = HiddenLayer(rng=numpy_rng,
                                        input=self.hidden_layers[-1].output,
                                        n_in=unrolled_hidden_layers_sizes[-1],
                                        n_out=n_ins, ###
                                        activation=T.nnet.sigmoid)

        self.params.extend(self.reconstructionLayer.params)


    def get_cost_updates(self, learning_rate): 
        """ This function computes the cost and the updates for one trainng
        step of the DA.
        *Note from dA tutorial: 'The reconstruction error can be measured in many
        ways, depending on the appropriate distributional assumptions on the 
        input given the code, e.g., using the traditional squared error, or if 
        the input is interpreted as either bit vectors or vectors of bit 
        probabilities by the reconstruction cross-entropy defined as (see L 
        below). 
        *Note from 'Getting Started' link: 'An image is represented as numpy 
        1-dimensional array of 784 (28 x 28) float values between 0 and 1 
        (0 stands for black, 1 for white). 
        http://www.deeplearning.net/tutorial/dA.html#daa
        http://www.deeplearning.net/tutorial/gettingstarted.html#gettingstarted
        """
        z = self.reconstructionLayer.output
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `DA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


    def build_finetune_functions(self, train_set_x, valid_set_x, test_set_x, 
                            batch_size): 
        ### removed learning_rate from function parameters
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        ### compute number of minibatches for validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')  ### added

        ### added
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size


        # train_fn = theano.function(inputs=[index],
        #       outputs=self.finetune_cost,
        #       updates=updates,
        #       givens={self.x: train_set_x[index * batch_size:
        #                                   (index + 1) * batch_size],
        #               self.y: train_set_y[index * batch_size:
        #                                   (index + 1) * batch_size]})

        # test_score_i = theano.function([index], self.errors,
        #          givens={self.x: test_set_x[index * batch_size:
        #                                     (index + 1) * batch_size],
        #                  self.y: test_set_y[index * batch_size:
        #                                     (index + 1) * batch_size]})

        # valid_score_i = theano.function([index], self.errors,
        #       givens={self.x: valid_set_x[index * batch_size:
        #                                   (index + 1) * batch_size],
        #               self.y: valid_set_y[index * batch_size:
        #                                   (index + 1) * batch_size]})

        # # Create a function that scans the entire validation set
        # def valid_score():
        #     return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # # Create a function that scans the entire test set
        # def test_score():
        #     return [test_score_i(i) for i in xrange(n_test_batches)]

        # return train_fn, valid_score, test_score
        ### end DLT code block
        


        # get the cost and the updates list
        cost, updates = self.get_cost_updates(learning_rate=learning_rate)
        
        # compile the theano function
        train_fn = theano.function(inputs=[index,
                          ###theano.Param(corruption_level, default=0.2),
                          theano.Param(learning_rate, default=0.1)],
                             outputs=cost,
                             updates=updates,
                             givens={self.x: train_set_x[batch_begin:
                                                         batch_end]})
        


        ### get cost for validation set 
        # valid_cost = autoencoder.get_cost_only()
        # v_fn = theano.function(inputs=[index],
        #                         outputs=valid_cost
        #                         givens={self.x: valid_set_x[batch_begin:
        #                                                     batch_end]})
        # valid_fns.append(v_fn)
        ### can repeat the previous 5 lines of code for test set cost

        


        # return (train_fns, valid_fns)
        return train_fn
        ### end jdy code block


def test_DAfinetune(finetune_lr=0.1, pretraining_epochs=1,
             pretrain_lr=0.1, k=1, training_epochs=1,
             dataset='/Users/jon/Data/mnist/mnist.pkl.gz', batch_size=10):
    ### finetune_lr and training_epochs not needed for SRBM

    """
    Demonstrates how to train and test a Deep Autoencoder.

    This is demonstrated on MNIST.

    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    :default: 0.1
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining 
    :default: 100
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :default: 0.01
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :default: 1
    :type training_epochs: int
    :param training_epochs: maximal number of iterations to run the optimizer. 
    :default: 1000
    :type dataset: string
    :param dataset: path the the pickled dataset
    :default: 'mnist.pkl.gz'
    :type batch_size: int
    :param batch_size: the size of a minibatch
    :default: 10
    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    #print n_train_batches

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print '... building the model'

    # construct the SRBM_SA 
    dafinetune = DAfinetune(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 500, 250, 30])

    ### jdy code block
    print dafinetune.params
    print 'layer0'
    print dafinetune.params[0].get_value()[0:3, 0:3]
    print 'layer1'
    print dafinetune.params[2].get_value()[0:3, 0:3]
    print 'layer2'
    print dafinetune.params[4].get_value()[0:3, 0:3]
    ###

    # ### *note
    # '''Now any function pretrain_fns[i] takes as arguments index and optionally 
    # lr - the learning rate. Note that the names of the parameters are the names
    # given to the Theano variables (e.g. lr) when they are constructed and not 
    # the python variables (e.g. learning_rate).'''
    

    # print '... pre-training the model'
    # start_time = time.time()  ###changed time.clock() to time.time() because
    # ### getting times that are way too long with time.clock()

    # ## Pre-train layer-wise
    # for i in xrange(srbm_sa.n_layers):
    #     # go through pretraining epochs
    #     for epoch in xrange(pretraining_epochs):
    #         # go through the training set
    #         c = []
    #         for batch_index in xrange(n_train_batches):
    #             c.append(pretraining_fns[i](index=batch_index,
    #                                         lr=pretrain_lr))
    #             ### see *note above

    #             ### for each function call of the pretraining fns, the states 
    #             ### of the shared variables (e.g. self.params) are updated.
    #             ### pretraining_fns = list of cost/update functions for each
    #             ### layer of SRBM_SA. Each call to this fxn returns the cost and 
    #             ### updates the parameters for that layer. See 'Shared Variable'  
    #             ### section here: http://deeplearning.net/software/theano/tutorial/examples.html#logistic-function
    #         print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
    #         print numpy.mean(c)

    #         ### jdy code block
    #         # print srbm_sa.params 
    #         # print 'layer %i, epoch %d' % (i,epoch)
    #         # jdy_params0 = srbm_sa.params[i * 2].get_value() 
    #         # print jdy_params0.shape
    #         # print jdy_params0[0:3, 0:3]
    #         ###
            

    # end_time = time.time()  ###changed time.clock() to time.time()
    # print >> sys.stderr, ('The pretraining code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # ###jdy code block
    # # print i, epoch, batch_index
    # # temp = pretraining_fns[i](index=batch_index,lr=pretrain_lr))
    # ### Running the line above changes the weights in layer 3 by a very small
    # ### amount leading to a minimal change in cost. any time pretraining_fns[i] 
    # ### is called it will update the shared variables for that layer only.
    # # print temp
    # # params = type(srbm_sa.params[0])
    # # print params


    # # print srbm_sa.params
    # # print 'layer0'
    # # print srbm_sa.params[0].get_value()[0:3, 0:3]
    # # print 'layer1'
    # # print srbm_sa.params[2].get_value()[0:3, 0:3]
    # # print 'layer2'
    # # print srbm_sa.params[4].get_value()[0:3, 0:3]
    # ###

    # # save_short(srbm_sa, '/Users/jon/models/DBNDA_theano/model_test.pkl')
    # # save_med_pkl(srbm_sa, '/Users/jon/models/DBNDA_theano/model_test2.pkl')
    # # save_med_npy(srbm_sa, '/Users/jon/models/DBNDA_theano/model_test3.npy')



    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'

    training_fn = dafinetune.build_finetune_functions(train_set_x=train_set_x,
                                                valid_set_x=valid_set_x, 
                                                test_set_x=test_set_x,
                                                batch_size=batch_size)

    print '... finetuning the model'
    start_time = time.time() 

    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(training_fn(index=batch_index, lr=pretrain_lr))

        print 'Training epoch %d, cost' % (epoch),
        print numpy.mean(c)

    end_time = time.time()
    print >> sys.stderr, ('The fine tuning code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time)
                                              / 60.))


    # ## Finetune train layer-wise
    # for layer in xrange(srbm_sa.n_layers):
    #     # go through training epochs
    #     for ep in xrange(training_epochs):
    #         # go through the training set
    #         t_cost = []
    #         for b_index in xrange(n_train_batches):
    #             t_cost.append(training_fns[layer](index=b_index,
    #                                         lr=finetune_lr))
    #             ### see *note above

    #             ### for each function call of the training fn, the states 
    #             ### of the shared variables (e.g. self.params) are updated.
    #             ### training_fn = one cost/update function for the DA. 
                  ### Each call to this fxn returns the cost and 
    #             ### updates the parameters for the entire model. See 'Shared Variable'  
    #             ### section here: http://deeplearning.net/software/theano/tutorial/examples.html#logistic-function
    #         print 'Pre-training layer %i, epoch %d, cost ' % (layer, ep),
    #         print numpy.mean(t_cost)

    #         ### jdy code block
    #         # print srbm_sa.params 
    #         # print 'layer %i, epoch %d' % (layer,ep)
    #         # jdy_params0train = srbm_sa.params[layer * 2].get_value() 
    #         # print jdy_params0train.shape
    #         # print jdy_params0train[0:3, 0:3]
    #         ###

    # e_time = time.time()  ###changed time.clock() to time.time()
    # print >> sys.stderr, ('The finetuning code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((e_time - s_time) / 60.))


    # ## jdy code block
    # print layer, ep, b_index
    # params = type(srbm_sa.params[0])
    # print params


    # print srbm_sa.params
    # print 'layer0'
    # print srbm_sa.params[0].get_value()[0:3, 0:3]
    # print 'layer1'
    # print srbm_sa.params[2].get_value()[0:3, 0:3]
    # print 'layer2'
    # print srbm_sa.params[4].get_value()[0:3, 0:3]
    # ##

    # train_fn, validate_model, test_model = srbm_sa.build_finetune_functions(
    #             datasets=datasets, batch_size=batch_size,
    #             learning_rate=finetune_lr)

    # print '... finetunning the model'
    # # early-stopping parameters
    # patience = 4 * n_train_batches  # look as this many examples regardless
    # patience_increase = 2.    # wait this much longer when a new best is
    #                           # found
    # improvement_threshold = 0.995  # a relative improvement of this much is
    #                                # considered significant
    # validation_frequency = min(n_train_batches, patience / 2)
    #                               # go through this many
    #                               # minibatche before checking the network
    #                               # on the validation set; in this case we
    #                               # check every epoch

    # best_params = None
    # best_validation_loss = numpy.inf
    # test_score = 0.
    # start_time = time.time()

    # done_looping = False
    # epoch = 0

    # while (epoch < training_epochs) and (not done_looping):
    #     epoch = epoch + 1
    #     for minibatch_index in xrange(n_train_batches):

    #         minibatch_avg_cost = train_fn(minibatch_index)
    #         iter = (epoch - 1) * n_train_batches + minibatch_index

    #         if (iter + 1) % validation_frequency == 0:

    #             validation_losses = validate_model()
    #             this_validation_loss = numpy.mean(validation_losses)
    #             print('epoch %i, minibatch %i/%i, validation error %f %%' % \
    #                   (epoch, minibatch_index + 1, n_train_batches,
    #                    this_validation_loss * 100.))

    #             # if we got the best validation score until now
    #             if this_validation_loss < best_validation_loss:

    #                 #improve patience if loss improvement is good enough
    #                 if (this_validation_loss < best_validation_loss *
    #                     improvement_threshold):
    #                     patience = max(patience, iter * patience_increase)

    #                 # save best validation score and iteration number
    #                 best_validation_loss = this_validation_loss
    #                 best_iter = iter

    #                 # test it on the test set
    #                 test_losses = test_model()
    #                 test_score = numpy.mean(test_losses)
    #                 print(('     epoch %i, minibatch %i/%i, test error of '
    #                        'best model %f %%') %
    #                       (epoch, minibatch_index + 1, n_train_batches,
    #                        test_score * 100.))

    #         if patience <= iter:
    #             done_looping = True
    #             break

    # end_time = time.time()
    # print(('Optimization complete with best validation score of %f %%,'
    #        'with test performance %f %%') %
    #              (best_validation_loss * 100., test_score * 100.))
    # print >> sys.stderr, ('The fine tuning code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time)
    #                                           / 60.))


if __name__ == '__main__':
    test_DAfinetune()
