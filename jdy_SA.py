"""
 ### this code is a modified version of SdA.py. (modified by jdy) 
 CHANGES:  updates imports. removes corruption_level, supervised portions. 
 changes dA to A and SdA to SA. removes self.logLayer. adds comments. comments
 out build_finetune_functions. adds print statements. updates time.clock to 
 time.time. comments out "Finetuning the Model" in test function.

 This tutorial introduces stacked auto-encoders (SA) using Theano.

 Autoencoders are the building blocks for SA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 References :
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

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

#from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
#from dA import dA

from jdy_A import A
from jdy_utils import load_data, save_short, save_med_pkl, save_med_npy


class SA(object):
    """Stacked auto-encoder class (SA)

    A stacked autoencoder model is obtained by stacking several
    As. The hidden layer of the A at layer `i` becomes the input of
    the A at layer `i+1`. The first layer A gets as input the input of
    the SA, and the hidden layer of the last A represents the output.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,):
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

        self.sigmoid_layers = []
        self.A_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        ###self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        ### comments from DLT                         
        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SA
            self.params.extend(sigmoid_layer.params)

            # Construct an autoencoder that shared weights with this
            # layer
            A_layer = A(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.A_layers.append(A_layer)

        # We now need to add a logistic layer on top of the MLP
        
        ###can remove this
        # self.logLayer = LogisticRegression(
        #                  input=self.sigmoid_layers[-1].output,
        #                  n_in=hidden_layers_sizes[-1], n_out=n_outs)

        # self.params.extend(self.logLayer.params)

        # # construct a function that implements one step of finetunining

        # # compute the cost for second phase of training,
        # # defined as the negative log likelihood
        # self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # # compute the gradients with respect to the model parameters
        # # symbolic variable that points to the number of errors made on the
        # # minibatch given by self.x and self.y
        # self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in training the A corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a A you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the A

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the A layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        ###corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        ###n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for autoencoder in self.A_layers:
            # get the cost and the updates list
            cost, updates = autoencoder.get_cost_updates(learning_rate=learning_rate)
            
            # compile the theano function
            ### Each function in this list of functions (pretrain_fns), computes
            ### the cost and updates (to the weights and biases) for that layer 
            ###individually(greedily).See the following link for how this works:
            ### http://www.toptal.com/machine-learning/an-introduction-to-deep-learning-from-perceptrons-to-deep-networks
            ### the input to each of these changes because of how the layer 
            ### input is defined above. To train greedily like this is going to
            ### take much longer than training with classic backpropagation, but
            ### it is supposed to work better (see link above).
            fn = theano.function(inputs=[index,
                              ###theano.Param(corruption_level, default=0.2),
                              theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_set_x[batch_begin:
                                                             batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    # def build_finetune_functions(self, datasets, batch_size, learning_rate):
    #     '''Generates a function `train` that implements one step of
    #     finetuning, a function `validate` that computes the error on
    #     a batch from the validation set, and a function `test` that
    #     computes the error on a batch from the testing set

    #     :type datasets: list of pairs of theano.tensor.TensorType
    #     :param datasets: It is a list that contain all the datasets;
    #                      the dataset has to contain three pairs, `train`,
    #                      `valid`, `test` in this order, where each pair
    #                      is formed of two Theano variables, one for the
    #                      datapoints, the other for the labels

    #     :type batch_size: int
    #     :param batch_size: size of a minibatch

    #     :type learning_rate: float
    #     :param learning_rate: learning rate used during finetune stage
    #     '''

    #     (train_set_x, train_set_y) = datasets[0]
    #     (valid_set_x, valid_set_y) = datasets[1]
    #     (test_set_x, test_set_y) = datasets[2]

    #     # compute number of minibatches for training, validation and testing
    #     n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    #     n_valid_batches /= batch_size
    #     n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    #     n_test_batches /= batch_size

    #     index = T.lscalar('index')  # index to a [mini]batch

    #     # compute the gradients with respect to the model parameters
    #     gparams = T.grad(self.finetune_cost, self.params)

    #     # compute list of fine-tuning updates
    #     updates = []
    #     for param, gparam in zip(self.params, gparams):
    #         updates.append((param, param - gparam * learning_rate))

    #     train_fn = theano.function(inputs=[index],
    #           outputs=self.finetune_cost,
    #           updates=updates,
    #           givens={
    #             self.x: train_set_x[index * batch_size:
    #                                 (index + 1) * batch_size],
    #             self.y: train_set_y[index * batch_size:
    #                                 (index + 1) * batch_size]},
    #           name='train')

    #     test_score_i = theano.function([index], self.errors,
    #              givens={
    #                self.x: test_set_x[index * batch_size:
    #                                   (index + 1) * batch_size],
    #                self.y: test_set_y[index * batch_size:
    #                                   (index + 1) * batch_size]},
    #                   name='test')

    #     valid_score_i = theano.function([index], self.errors,
    #           givens={
    #              self.x: valid_set_x[index * batch_size:
    #                                  (index + 1) * batch_size],
    #              self.y: valid_set_y[index * batch_size:
    #                                  (index + 1) * batch_size]},
    #                   name='valid')

    #     # Create a function that scans the entire validation set
    #     def valid_score():
    #         return [valid_score_i(i) for i in xrange(n_valid_batches)]

    #     # Create a function that scans the entire test set
    #     def test_score():
    #         return [test_score_i(i) for i in xrange(n_test_batches)]

    #     return train_fn, valid_score, test_score


def test_SA(finetune_lr=999, pretraining_epochs=1,
             pretrain_lr=0.01, training_epochs=999,
             dataset='/Users/jon/Data/mnist/mnist.pkl.gz', batch_size=10):
    ### using batch size = 10 (instead of batch size = 1) leads to ~10% higher 
    ### cost values at each epoch. But, batch size = 10 runs much faster.
    """
    Demonstrates how to train and test an autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked autoencoder class
    sa = SA(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)

    ### jdy code block
    print sa.params
    print 'layer0'
    print sa.params[0].get_value()[0:3, 0:3]
    print 'layer1'
    print sa.params[2].get_value()[0:3, 0:3]
    print 'layer2'
    print sa.params[4].get_value()[0:3, 0:3]
    ###

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sa.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.time()  ###changed time.clock() to time.time()
    ## Pre-train layer-wise
    ###corruption_levels = [.1, .2, .3]
    for i in xrange(sa.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         ###corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

            ### jdy code block
            print sa.params 
            print 'layer %i, epoch %d' % (i,epoch)
            jdy_params0 = sa.params[i * 2].get_value() 
            print jdy_params0.shape
            print jdy_params0[0:3, 0:3]
            ###

    end_time = time.time()  ###changed time.clock() to time.time()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    ### jdy code block
    print sa.params
    print 'layer0'
    print sa.params[0].get_value()[0:3, 0:3]
    print 'layer1'
    print sa.params[2].get_value()[0:3, 0:3]
    print 'layer2'
    print sa.params[4].get_value()[0:3, 0:3]
    ###

    ########################
    # FINETUNING THE MODEL #
    ########################

    # # get the training, validation and testing function for the model
    # print '... getting the finetuning functions'
    # train_fn, validate_model, test_model = sda.build_finetune_functions(
    #             datasets=datasets, batch_size=batch_size,
    #             learning_rate=finetune_lr)

    # print '... finetunning the model'
    # # early-stopping parameters
    # patience = 10 * n_train_batches  # look as this many examples regardless
    # patience_increase = 2.  # wait this much longer when a new best is
    #                         # found
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
    # start_time = time.clock()

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
    #             print('epoch %i, minibatch %i/%i, validation error %f %%' %
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

    # end_time = time.clock()
    # print(('Optimization complete with best validation score of %f %%,'
    #        'with test performance %f %%') %
    #              (best_validation_loss * 100., test_score * 100.))
    # print >> sys.stderr, ('The training code for file ' +
    #                       os.path.split(__file__)[1] +
    #                       ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_SA()
