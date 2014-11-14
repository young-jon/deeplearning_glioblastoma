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

# from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from rbm import RBM

from jdy_utils import load_data, save_short, save_med_pkl, save_med_npy


class SRBM(object): 
    """Stacked Restricted Boltzmann Machine

    A stacked Restricted Boltzmann Machine is obtained by stacking several RBMs 
    on top of each other. The hidden layer of the RBM at layer `i` becomes the 
    input of the RBM at layer `i+1`. The first layer RBM gets as input the input
    of the network, and the hidden layer of the last RBM represents the output. 

    A SRBM is only the pretraining step of a Deep Belief Network (DBN). This 
    assumes the definition of DBN according to Deep Learning Tutorials (DLT). 

    DBN definition according to DLT:
    A Deep Belief Network is obtained by stacking several RBMs on top of each 
    other. The hidden layer of the RBM at layer `i` becomes the input of the RBM
    at layer `i+1`. The first layer RBM gets as input the input of the network, 
    and the hidden layer of the last RBM represents the output. When used for 
    classification, the DBN is treated as a MLP, by adding a logistic regression
    layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the SRBM

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []  ###think of as a list of hidden layers to make
        ### it easier to get input for each rbm.
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        ###self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels
                                 ### delete for SRBM-SA

        #from DLT
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to changing the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the SRBM-SA if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output  
                ### could change to self.rbm_layers[-1].propup. then i wouldn't 
                ### have to create self.sigmoid_layers

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
                                        ### dA_layer

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer) 

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the SRBM-SA. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the SRBM-SA.
            self.params.extend(sigmoid_layer.params) 

            # Construct an RBM that shared weights with this layer
            ### for input could also use self.propup (rbm.propup), but would 
            ### have to change additional code. layer_input here is is just 
            ### sigmoid(T.dot()) of previous layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,  ###activations of previous layer
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,  ###HiddenLayer class initializes
                            hbias=sigmoid_layer.b)  
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        ###can remove this
        # self.logLayer = LogisticRegression(
        #     input=self.sigmoid_layers[-1].output,
        #     n_in=hidden_layers_sizes[-1],
        #     n_out=n_outs)
        # self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        # self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        ### from logistic_sgd.py, self.negative_log_likelihood(self, y):
            ### return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
            ### this is a cost function given in terms of all the params of the
            ### model because you need to forward propagate through all layers
            ### to get the input for the function .p_y_given_x

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        # self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates, ### how to update shared vars.
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

        ###deleted def build_finetune_functions


def test_SRBM(finetune_lr=999, pretraining_epochs=2,
             pretrain_lr=0.01, k=1, training_epochs=999,
             dataset='/Users/jon/Data/mnist/mnist.pkl.gz', batch_size=10):
    ### finetune_lr and training_epochs not needed for SRBM

    """
    Demonstrates how to train and test a Deep Belief Network.

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

    # construct the Stacked RBMs
    srbm = SRBM(numpy_rng=numpy_rng, n_ins=28 * 28,
              hidden_layers_sizes=[500, 250, 100],
              n_outs=10)

    ### jdy code block
    print srbm.params
    print 'layer0'
    print srbm.params[0].get_value()[0:3, 0:3]
    print 'layer1'
    print srbm.params[2].get_value()[0:3, 0:3]
    print 'layer2'
    print srbm.params[4].get_value()[0:3, 0:3]
    ###

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    ### creates a list of pretraining fxns for each layer in the SRBM. This is
    ### where the self.sigmoid_layer[-1].output is needed -- to create the 
    ### appropriate equation/function for pretraining
    pretraining_fns = srbm.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size,
                                                k=k)

    ### *note
    '''Now any function pretrain_fns[i] takes as arguments index and optionally 
    lr - the learning rate. Note that the names of the parameters are the names
    given to the Theano variables (e.g. lr) when they are constructed and not 
    the python variables (e.g. learning_rate).'''
    

    print '... pre-training the model'
    start_time = time.time()
    ## Pre-train layer-wise
    for i in xrange(srbm.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))
                ### see *note above

                ### for each function call of the pretraining fns, the states 
                ### of the shared variables (e.g. self.params) are updated.
                ### pretraining_fns = list of cost/update functions for each
                ### layer of SRBM. Each call to this fxn returns the cost and 
                ### updates the parameters for that layer. See 'Shared Variable'  
                ### section here: http://deeplearning.net/software/theano/tutorial/examples.html#logistic-function
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

            ### jdy code block
            # print srbm.params 
            # print 'layer %i, epoch %d' % (i,epoch)
            # jdy_params0 = srbm.params[i * 2].get_value() 
            # print jdy_params0.shape
            # print jdy_params0[0:3, 0:3]
            ###

    end_time = time.time()
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ###jdy code block
    # print i, epoch, batch_index
    # temp = pretraining_fns[i](index=batch_index,lr=pretrain_lr))
    ### Running the line above changes the weights in layer 3 by a very small
    ### amount leading to a minimal change in cost. any time pretraining_fns[i] 
    ### is called it will update the shared variables for that layer only.
    # print temp
    # params = type(srbm_sa.params[0])
    # print params


    # print srbm.params
    # print 'layer0'
    # print srbm.params[0].get_value()[0:3, 0:3]
    # print 'layer1'
    # print srbm.params[2].get_value()[0:3, 0:3]
    # print 'layer2'
    # print srbm.params[4].get_value()[0:3, 0:3]
    # ###

    # save_short(srbm, '/Users/jon/models/DBNDA_theano/model_test.pkl')
    # save_med_pkl(srbm, '/Users/jon/models/DBNDA_theano/model_test2.pkl')
    # save_med_npy(srbm, '/Users/jon/models/DBNDA_theano/model_test3.npy')

    ###deleted 'finetuning the model'


if __name__ == '__main__':
    test_SRBM()
