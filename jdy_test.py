###TESTING###
import sys
import numpy
import theano

import jdy_A

from jdy_utils import load_data, load_med_pkl


def test_jdy_A():
	'''
	rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    n_hidden=500
    '''
	a = jdy_A.test_A(learning_rate=0.1, training_epochs=2, 
					dataset='/Users/jon/Data/mnist/mnist.pkl.gz',
					batch_size=20, output_folder='dA_plots', testing=1)

	model = load_med_pkl('/Users/jon/Models/DBNDA_theano/jdy_A_test_output.pkl', 
							3)
	print
	print '***TESTING***'
	print
	test0 = a.params[0].get_value()==model[0]
	print False not in test0
	test1 = a.params[1].get_value()==model[1]
	print False not in test1
	test2 = a.params[2].get_value()==model[2]
	print False not in test2


def test_Srbm_DAfinetune(module, obj):
	'''all tests should evaluate to true (on home mac) if the params below
	are use to initiate run_Srbm_DAfinetune. will need to update values in this
	function if want to use on work mac.

	pretraining_epochs=1, training_epochs=5, 
	hidden_layers_sizes=[1000, 500, 250, 30], 
	finetune_lr=0.1, pretrain_lr=0.1, 
	k=1, batch_size=10, 
	dataset='/Users/jon/Data/mnist/mnist.pkl.gz'

	### need to make sure this randomstate is used:
	numpy_rng = numpy.random.RandomState(123)'''
	if module == 1:
		###TESTING1 GOOD
		a=numpy.array([[ 0.09879994, -0.03318569,  0.08856042],
		[ 0.02100114,  0.05656936,  0.07261958],
		[ 0.20698177,  0.14437415,  0.20790022]])
		print (a==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '1'

		b=numpy.array([[ 0.0544301,  -0.25127706,  0.22170671],
		[ 0.15351117, -0.2231685,   0.1926219 ],
		[-0.08054012,  0.21709459,  0.03015934]] )
		print (b==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '2'
			
		c=numpy.array([[-0.20517612,  0.18863321,  0.30801029],
		[ 0.14593372,  0.29862848,  0.09555362],
		[-0.19470544,  0.01221362, -0.31747911]])
		print (c==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '3'

		print 0.0==numpy.round(obj.params[3].get_value()[0],2), '4'

		print (a==numpy.round(obj.rbm_params[0].get_value()[0:3, 0:3], 8)).all(), '5'

		print (b==numpy.round(obj.rbm_params[3].get_value()[0:3, 0:3], 8)).all(), '6'

		print (c==numpy.round(obj.rbm_params[6].get_value()[0:3, 0:3], 8)).all(), '7'

		print 0.0==numpy.round(obj.rbm_params[1].get_value()[0],2), '8'
	 	###END TESTING1

	 	### TESTING1 BAD. ALL TESTS FAIL
		# a=numpy.array([[ 0.09879994, -0.03318569,  0.08856043],
		# [ 0.02100114,  0.05656936,  0.07261958],
		# [ 0.20698177,  0.14437415,  0.20790022]])
		# print (a==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '1'

		# b=numpy.array([[ 0.0544301,  -0.25127706,  0.22170671],
		# [ 0.15351117, -0.2131685,   0.1926219 ],
		# [-0.08054012,  0.21709459,  0.03015934]] )
		# print (b==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '2'

		# c=numpy.array([[-0.20517612,  0.18863321,  0.30801029],
		# [ 0.14593372,  0.29861848,  0.09555362],
		# [-0.19470544,  0.01221362, -0.31747911]])
		# print (c==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '3'

		# print 0.1==numpy.round(obj.params[3].get_value()[0],2), '4'

		# print (a==numpy.round(obj.rbm_params[0].get_value()[0:3, 0:3], 8)).all(), '5'

		# print (b==numpy.round(obj.rbm_params[3].get_value()[0:3, 0:3], 8)).all(), '6'

		# print (c==numpy.round(obj.rbm_params[6].get_value()[0:3, 0:3], 8)).all(), '7'

		# print 0.8==numpy.round(obj.rbm_params[1].get_value()[0],2), '8'
 		###END TESTING1

 	if module == 2:
 		###TESTING2 GOOD
		d=numpy.array([[ 0.00889808, -0.12697725,  0.01510663],
		[-0.06354312,  0.00103337,  0.02840916],
		[ 0.12320258,  0.09890467,  0.13424923]])
		print (d==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '10'

		e=numpy.array([[ 0.48946507, -0.32782729,  0.08150482],
		[ 0.08153848, -0.18980341,  0.15371413],
		[-0.18342055, -0.27116513, -0.04994315]] )
		print (e==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '11'

		f=numpy.array([[-0.24739512,  0.20979701,  0.1809979 ],
		[ 0.04023768,  0.02864404, -0.40347024],
		[-0.27282463, -0.15462937, -0.22956235]])
		print (f==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '12'

		print -0.507665766==numpy.round(obj.params[3].get_value()[0],9), '13'

		print (d==numpy.round(obj.rbm_params[0].get_value()[0:3, 0:3], 8)).all(), '14'

		print (e==numpy.round(obj.rbm_params[3].get_value()[0:3, 0:3], 8)).all(), '15'

		print (f==numpy.round(obj.rbm_params[6].get_value()[0:3, 0:3], 8)).all(), '16'

		print -1.078679724==numpy.round(obj.rbm_params[5].get_value()[0],9), '17'
	 	###END TESTING2

 		###TESTING2 BAD. ALL TESTS FAIL
 	# 	d=numpy.array([[ 0.00889808, -0.12697725,  0.01510663],
		# [-0.06354312,  0.00103237,  0.02840916],
		# [ 0.12320258,  0.09890467,  0.13424923]])
		# print (d==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '10'

		# e=numpy.array([[ 0.48946507, -0.32782729,  0.08150482],
		# [ 0.08153848, -0.18980341,  0.15371412],
		# [-0.18342055, -0.27116513, -0.04994315]] )
		# print (e==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '11'

		# f=numpy.array([[-0.24739512,  0.20979701,  0.1809979 ],
		# [ 0.04023768,  0.02864404, -0.40347024],
		# [-0.27282463, -0.15462937, -0.22956239]])
		# print (f==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '12'

		# print 0.507665766==numpy.round(obj.params[3].get_value()[0],9), '13'

		# print (d==numpy.round(obj.rbm_params[0].get_value()[0:3, 0:3], 8)).all(), '14'

		# print (e==numpy.round(obj.rbm_params[3].get_value()[0:3, 0:3], 8)).all(), '15'

		# print (f==numpy.round(obj.rbm_params[6].get_value()[0:3, 0:3], 8)).all(), '16'

		# print -1.078679704==numpy.round(obj.params[5].get_value()[0],9), '17'
	 	###END TESTING2
	
	if module == 3:
		###TESTING3 GOOD
		g=numpy.array([[ 0.00889808, -0.12697725,  0.01510663],
		[-0.06354312,  0.00103337,  0.02840916],
		[ 0.12320258,  0.09890467,  0.13424923]])
		print (g==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '18'

		h=numpy.array([[ 0.48946507, -0.32782729,  0.08150482],
		[ 0.08153848, -0.18980341,  0.15371413],
		[-0.18342055, -0.27116513, -0.04994315]])
		print (h==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '19'

		j=numpy.array([[-0.24739512,  0.20979701,  0.1809979 ],
		[ 0.04023768,  0.02864404, -0.40347024],
		[-0.27282463, -0.15462937, -0.22956235]])
		print (j==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '20'
		###END TESTING3

		###TESTING3 BAD
		# g=numpy.array([[ 0.00889808, -0.12697725,  0.01510663],
		# [-0.06354312,  0.00103337,  0.02840916],
		# [ 0.19320258,  0.09890467,  0.13424923]])
		# print (g==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '18'

		# h=numpy.array([[ 0.48946506, -0.32782729,  0.08150482],
		# [ 0.08153848, -0.18980341,  0.15371413],
		# [-0.18342055, -0.27116513, -0.04994315]])
		# print (h==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '19'

		# j=numpy.array([[-0.24739512,  0.20979701,  0.1809979 ],
		# [ 0.04023768,  0.02864404, -0.40347024],
		# [-0.27282463, 0.15462937, -0.22956235]])
		# print (j==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '20'
		###END TESTING3


if __name__ == '__main__':
    test_jdy_A()