import cPickle
import gzip
import os
import sys
import time

import numpy

from jdy_Srbm import SRBM
from jdy_DAfinetune import DAfinetune
from jdy_utils import load_data, save_short, save_med_pkl, save_med_npy, get_reconstructions
from jdy_visualize import create_images


def preprocess_pretrain_params(object_params):
	#maybe put this in jdy_utils
	'''
	creates a list of non transposed weights and biases (in the correct 
	order) to be used when the pretraining parameters are used to initialize a 
	Deep Autoencoder. This makes it easier to loop through the paramaters when 
	creating hidden layers. Note that the decoder weights returned here still 
	need to be transposed before they are used to initialize a Deep Autoencoder.

	object_params: is a list of the pretraining parameters (self.rbm_params)
		-for a 4 layered network:
		-object_params = [W, b, vbias, W, b, vbias, W, b, vbias, W, b, vbias]

	sample return: 	weights = [W1,W2,W3,W4,W4,W3,W2,W1]
					biases = [b1,b2,b3,b4,vbias4,vbias3,vbias2,vbias1]
	'''

	length = len(object_params)
	encoder_weights = []
	decoder_weights = []
	hbiases = []
	vbiases = []
	for i in xrange(0, length, 3):
		encoder_weights.append(object_params[i])
	for j in xrange(1, length+1, 3):
		hbiases.append(object_params[j])
	for k in xrange(2, length+1, 3):
		vbiases.append(object_params[k])

	### NOTE: will still need to take the transpose of the decoder_weights later
	decoder_weights = encoder_weights[:]
	decoder_weights.reverse()
	weights = encoder_weights + decoder_weights

	vbiases.reverse()
	biases = hbiases + vbiases

	return weights, biases

def run_Srbm_DAfinetune(pretraining_epochs=1, training_epochs=5, 
						hidden_layers_sizes=[1000, 500, 250, 30],
						finetune_lr=0.1, pretrain_lr=0.1, 
						k=1, batch_size=10, 
						dataset='/Users/jon/Data/mnist/mnist.pkl.gz',
						image_finetune_epochs=[0,2,4]):

	'''image_finetune_epochs = finetune epochs after which we create an image'''

	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	n_ins = train_set_x.get_value(borrow=True).shape[1]

	# compute number of minibatches for training, validation and testing
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	# numpy random generator
	numpy_rng = numpy.random.RandomState(123)


	#########################
	# PRETRAINING THE MODEL #
	#########################

	print '... building the Stacked RBMs model'
	# construct the Stacked RBMs
	srbm = SRBM(numpy_rng=numpy_rng, n_ins=n_ins, 
				hidden_layers_sizes=hidden_layers_sizes, n_outs=10)
			### calculate n_ins from train_set_x so isn't hard coded in SRBM 
			### params or pass it in at run_Srbm_DAfinetune. Also can get rid of 
			### n_outs=10. this is only used for supervised learning. for 
			### supervised learning just use DLT's neural network implementation





	###IMAGES  START JDY CODE BLOCK
	###it is hard-coded for for all images to have row1=input data, 
	###row2=random initialization, and row3=after pretraining is complete.

	### Setup Images Environment
	image_dataset = train_set_x
	n_samples = image_dataset.get_value(borrow=True).shape[0] 
	reconst_len = n_ins   #true for unsupervised learning
	n_columns = 20
	total_num_images = (len(image_finetune_epochs)+3) * n_columns

	### create all_images to store final input data and reconstructions to be 
	### displayed as a tiled image once the algorithm completes
	all_images=numpy.zeros((total_num_images, reconst_len))
	all_images[0:n_columns] = image_dataset.get_value()[311:331] 
	###change to borrow=True? prob okay since I am never changing this shared 
	###variable by side effect
	#image_row_counter = 1  # to keep track of image rows for later indexing

	### Create DAfinetune object and initial reconstructions using random 
	### weights for images
	weights, biases = preprocess_pretrain_params(srbm.rbm_params)
	pretrain_unrolled = DAfinetune(numpy_rng=numpy_rng, n_ins=n_ins, 
								weights=weights, biases=biases, 
								hidden_layers_sizes=hidden_layers_sizes)

	r_2d = get_reconstructions(obj=pretrain_unrolled, data=image_dataset, 
								batch_size=batch_size, n_samples=n_samples, 
								reconst_len=reconst_len)

	# reconstruction_fn = pretrain_unrolled.build_reconstruction_function(
	# 												data=train_set_x,
	# 												batch_size=batch_size)
	# r=[]  #to store reconstructions
	# for batch_index in xrange(n_train_batches):
	# 	reconstructions = reconstruction_fn(index=batch_index)
	# 	r.append(reconstructions)
	print '******************************************************************'

	all_images[1*n_columns:2*n_columns] = r_2d[311:331]

	###END TODO JDY CODE BLOCK




	### jdy code block
	print srbm.params
	print 'layer0'
	print srbm.params[0].get_value()[0:3, 0:3], srbm.params[1].get_value()[0]
	print 'layer1'
	print srbm.params[2].get_value()[0:3, 0:3], srbm.params[3].get_value()[0]
	print 'layer2'
	print srbm.params[4].get_value()[0:3, 0:3], srbm.params[5].get_value()[0]
	print ''

	print srbm.rbm_params
	print 'layer0'
	print srbm.rbm_params[0].get_value()[0:3, 0:3], type(srbm.rbm_params[1].get_value()[0]), srbm.rbm_params[2].get_value()[0]
	print 'layer1'
	print srbm.rbm_params[3].get_value()[0:3, 0:3], srbm.rbm_params[4].get_value()[0], srbm.rbm_params[5].get_value()[0]
	print 'layer2'
	print srbm.rbm_params[6].get_value()[0:3, 0:3], srbm.rbm_params[7].get_value()[0], srbm.rbm_params[8].get_value()[0]

	# print ''

	# print srbm.rbm_params[9].get_value().T
	# print srbm.rbm_params[9].get_value().shape
	# print srbm.rbm_params[9].get_value().T.shape
	# print srbm.rbm_params[0].get_value().shape
	# print srbm.rbm_params[0].get_value().T.shape

	###

	print '... getting the pretraining functions'
	### creates a list of pretraining fxns for each layer in the SRBM. This is
	### where the self.sigmoid_layer[-1].output is needed -- to create the 
	### appropriate equation/function for pretraining
	pretraining_fns = srbm.pretraining_functions(train_set_x=train_set_x,
	                                            batch_size=batch_size,
	                                            k=k)

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
	            
	        print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
	        print numpy.mean(c)

	end_time = time.time()
	print >> sys.stderr, ('The pretraining code for file ' +
	                      os.path.split(__file__)[1] +
	                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

	### jdy code block
	print srbm.params
	print 'layer0'
	print srbm.params[0].get_value()[0:3, 0:3], srbm.params[1].get_value()[0]
	print 'layer1'
	print srbm.params[2].get_value()[0:3, 0:3], srbm.params[3].get_value()[0]
	print 'layer2'
	print srbm.params[4].get_value()[0:3, 0:3], srbm.params[5].get_value()[0]
	print ''

	print srbm.rbm_params
	print 'layer0'
	print srbm.rbm_params[0].get_value()[0:3, 0:3], srbm.rbm_params[1].get_value()[0], srbm.rbm_params[2].get_value()[0]
	print 'layer1'
	print srbm.rbm_params[3].get_value()[0:3, 0:3], srbm.rbm_params[4].get_value()[0], srbm.rbm_params[5].get_value()[0]
	print 'layer2'
	print srbm.rbm_params[6].get_value()[0:3, 0:3], srbm.rbm_params[7].get_value()[0], srbm.rbm_params[8].get_value()[0]
	###

	########################
	# FINETUNING THE MODEL #
	########################

	#get weights and biases from pretraining to use to initialize a Deep 
	#Autoencoder
	weights, biases = preprocess_pretrain_params(srbm.rbm_params)

	print '... building the Deep Autoencoder model'
	# construct the Deep Autoencoder 
	dafinetune = DAfinetune(numpy_rng=numpy_rng, n_ins=n_ins, 
				weights=weights, biases=biases, 
				hidden_layers_sizes=hidden_layers_sizes)

	###START JDY CODE BLOCK FOR IMAGES
	###TODO: AFTER PRETRAINING: create reconstructions and add to all_images.
	###The weights passed to DAfinetune here are the weights immediately after pretraining

	reconstruction_fn = dafinetune.build_reconstruction_function(
												data=train_set_x,
												batch_size=batch_size)
	r=[]  #to store reconstructions
	for batch_index in xrange(n_train_batches):
		reconstructions = reconstruction_fn(index=batch_index)
		r.append(reconstructions)

	r_3d_ndarray = numpy.asarray(r)
	r_2d_ndarray = r_3d_ndarray.reshape(50000,784)
	all_images[40:60] = r_2d_ndarray[311:331]
	print 'HERE epoch 1asdfasdf'

	###END JDY CODE BLOCK




	### jdy code block
	print dafinetune.params
	print 'layer0'
	print dafinetune.params[0].get_value()[0:3, 0:3]
	print 'layer1'
	print dafinetune.params[2].get_value()[0:3, 0:3]
	print 'layer2'
	print dafinetune.params[4].get_value()[0:3, 0:3]
	###

	# get the training, validation and testing function for the model
	print '... getting the finetuning functions'

	training_fn, testing_fn = dafinetune.build_finetune_functions(
	                                            train_set_x=train_set_x,
	                                            valid_set_x=valid_set_x, 
	                                            test_set_x=test_set_x,
	                                            batch_size=batch_size)

	print '... finetuning the model'
	start_time = time.time() 

	###ADDED to keep track of how many image rows have been added to all_images.
	### If I change the number of rows added to all_images will need to update 
	### image_row_counter.
	image_row_counter = 3

	# for each epoch
	for epoch in xrange(training_epochs):

	    c = []  #list to collect costs
	    e = []  #list to collect train errors
	    # TRAIN SET COST: for each batch, append the cost for that batch (should 
	    # be 5000 costs in c). Also append mse to e.
	    for batch_index in xrange(n_train_batches):
	        cost, err = training_fn(index=batch_index, lr=finetune_lr)
	        c.append(cost)
	        e.append(err)


	    # TEST SET ERROR: calculate test set error for each epoch
	    te = []  #list to collect test errors
	    for i in xrange(n_test_batches):
	        te.append(testing_fn(index=i))

	    # print results to screen
	    print ('Training epoch %d, Train cost %0.3f, Train MSE %0.3f, Test '
	        'MSE %0.3f' % (epoch, numpy.mean(c), numpy.mean(e), numpy.mean(te)))
	    

	    ###START JDY CODE BLOCK IMAGES
		### append reconstructions using parameters from after 'epoch'
	    
	    if epoch in image_finetune_epochs:
	    	#get_reconstructions(dafinetune, train_set_x, batch_size)
	    	###add_reconstructions(data, batch_size,)
			reconstruction_fn = dafinetune.build_reconstruction_function(
												data=train_set_x,
												batch_size=batch_size)
			r=[]  #to store reconstructions
			for batch_index in xrange(n_train_batches):
				reconstructions = reconstruction_fn(index=batch_index)
				r.append(reconstructions)

			r_3d_ndarray = numpy.asarray(r)
			r_2d_ndarray = r_3d_ndarray.reshape(50000,784)
			all_images[image_row_counter*n_columns:(image_row_counter+1)*n_columns] = r_2d_ndarray[311:331]
			image_row_counter += 1
			print 'HERE epoch 9999999999999999999999999999999999999999'

	### END JDY CODE BLOCK IMAGES

	create_images(all_images, image_row_counter, n_columns)



	    
	        
	end_time = time.time()
	print >> sys.stderr, ('The fine tuning code for file ' +
	                      os.path.split(__file__)[1] +
	                      ' ran for %.2fm' % ((end_time - start_time)
	                                          / 60.))


# def test_Srbm_DAfinetune():
# 	'''this test should give the output in evernote 'research2' 2/4/15. 
# 	this function does not currently work. need to write code to print True if 
# 	final state of the model (e.g. Test MSE == 9.077). would be nice to test a
# 	sampling of intermediate weights and errors as well...'''
# 	srbm = run_Srbm_DAfinetune(pretraining_epochs=5, training_epochs=5, 
# 						hidden_layers_sizes=[1000, 500, 250, 30],
# 						finetune_lr=0.1, pretrain_lr=0.1, 
# 						k=1, batch_size=10, 
# 						dataset='/Users/jdy10/Data/mnist/mnist.pkl.gz'):
# 	### need to make sure this randomstate is used:
# 	numpy_rng = numpy.random.RandomState(123)


if __name__ == '__main__':
	srbm = run_Srbm_DAfinetune()