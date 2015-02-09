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


def test_Srbm_DAfinetune(module, obj, computer, testing=None):
	'''all tests should evaluate to true (assuming correct computer is passed 
	in) if the params below are used to initiate run_Srbm_DAfinetune. 

	pretraining_epochs=1, training_epochs=5, 
	hidden_layers_sizes=[1000, 500, 250, 30], 
	finetune_lr=0.1, pretrain_lr=0.1, 
	k=1, batch_size=10, 
	dataset='/Users/jon/Data/mnist/mnist.pkl.gz'

	### need to make sure this randomstate is used:
	numpy_rng = numpy.random.RandomState(123)'''

	if computer == 'home':
		pretrain_cost=[-90.591655313, -125.134391165, -64.546611631, -54.767537662]
		d=numpy.array([[ 0.00889808, -0.12697725,  0.01510663],
		[-0.06354312,  0.00103337,  0.02840916],
		[ 0.12320258,  0.09890467,  0.13424923]])
		e=numpy.array([[ 0.48946507, -0.32782729,  0.08150482],
		[ 0.08153848, -0.18980341,  0.15371413],
		[-0.18342055, -0.27116513, -0.04994315]] )
		f=numpy.array([[-0.24739512,  0.20979701,  0.1809979 ],
		[ 0.04023768,  0.02864404, -0.40347024],
		[-0.27282463, -0.15462937, -0.22956235]])
		bias2 = -0.507665766
		bias3 = -1.078679724
		train_cost_error=[98.679,13.718,88.491,11.641,83.629,11.187,80.685,11.094,78.680,9.939]

		### TESTING BAD. ALL TESTS USING THESE VALUES FAIL
		# pretrain_cost=[-90.591655313, -125.134391165, -64.546611631, -54.7675376621]
		# d=numpy.array([[ 0.00889808, -0.12697725,  0.01510663],
		# [-0.06354312,  0.00103237,  0.02840916],
		# [ 0.12320258,  0.09890467,  0.13424923]])
		# e=numpy.array([[ 0.48946507, -0.32782729,  0.08150482],
		# [ 0.08153848, -0.18980341,  0.15371412],
		# [-0.18342055, -0.27116513, -0.04994315]] )
		# f=numpy.array([[-0.24739512,  0.20979701,  0.1809979 ],
		# [ 0.04023768,  0.02864404, -0.40347024],
		# [-0.27282463, -0.15462937, -0.22956239]])
		# bias2=0.507665766
		# bias3=-1.078679704
		# train_cost_error=[98.679,13.718,88.491,11.641,83.629,11.187,81.685,11.094,78.680,9.939]

	elif computer == 'work':
		pretrain_cost=[-90.594557945, -124.912082856, -65.339939832, -54.759187849]
		d=numpy.array([[ 0.00701483, -0.12709004,  0.01438415],
		[-0.06454756,  0.0008675,   0.02781272],
		[ 0.11075955, 0.09836527,  0.13550725]])
		e=numpy.array([[ 0.45034904, -0.24139931,  0.08253685],
		[ 0.06978032, -0.17375246,  0.16156176],
		[-0.18585386, -0.20028744, -0.04922392]])
		f=numpy.array([[-0.26877842, -0.04553986,  0.21810745],
		[ 0.04598604, -0.18929942, -0.38212723],
		[-0.2534254,  -0.18547121, -0.35068252]])
		bias2 = -0.469769795
		bias3 = -1.163395969
		train_cost_error=[98.462,13.805,88.568,11.918,84.432,12.144,81.187,10.166,78.375,9.708]

		###TESTING BAD. ALL TESTS USING THESE VALUES FAIL
		# pretrain_cost=[-90.594557945, -124.912082866, -65.339939832, -54.759087849]
		# d=numpy.array([[ 0.00701483, -0.12709004,  0.01438415],
		# [-0.06454756,  0.008675,   0.02781272],
		# [ 0.11075955, 0.09836527,  0.13550725]])
		# e=numpy.array([[ 0.45034904, -0.24139931,  0.08253685],
		# [ 0.06978031, -0.17375246,  0.16156176],
		# [-0.18585386, -0.20028744, -0.04922392]])
		# f=numpy.array([[-0.26877842, -0.04553986,  0.21810745],
		# [ 0.04598604, -0.18929942, -0.38212723],
		# [-0.2534254,  -0.18547121, 0.35068252]])
		# bias2 = -0.469769793
		# bias3 = -1.163395968
		# train_cost_error=[98.462,13.805,88.568,11.918,84.432,12.144,81.187,10.166,78.375,9.707]


	if module == 1:
		### Random Initialization is the same on both computers
		a=numpy.array([[ 0.09879994, -0.03318569,  0.08856042],
		[ 0.02100114,  0.05656936,  0.07261958],
		[ 0.20698177,  0.14437415,  0.20790022]])
		b=numpy.array([[ 0.0544301,  -0.25127706,  0.22170671],
		[ 0.15351117, -0.2231685,   0.1926219 ],
		[-0.08054012,  0.21709459,  0.03015934]] )
		c=numpy.array([[-0.20517612,  0.18863321,  0.30801029],
		[ 0.14593372,  0.29862848,  0.09555362],
		[-0.19470544,  0.01221362, -0.31747911]])
		bias = 0.0

		### TESTING1 BAD. ALL TESTS USING THESE VALUES FAIL
		# a=numpy.array([[ 0.09879994, -0.03318569,  0.08856043],
		# [ 0.02100114,  0.05656936,  0.07261958],
		# [ 0.20698177,  0.14437415,  0.20790022]])
		# b=numpy.array([[ 0.0544301,  -0.25127706,  0.22170671],
		# [ 0.15351117, -0.2131685,   0.1926219 ],
		# [-0.08054012,  0.21709459,  0.03015934]] )
		# c=numpy.array([[-0.20517612,  0.18863321,  0.30801029],
		# [ 0.14593372,  0.29861848,  0.09555362],
		# [-0.19470544,  0.01221362, -0.31747911]])
		# bias = 0.1

		print (a==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '1'
		print (b==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '2'
		print (c==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '3'
		print bias==numpy.round(obj.params[3].get_value()[0],2), '4'
		print (a==numpy.round(obj.rbm_params[0].get_value()[0:3, 0:3], 8)).all(), '5'
		print (b==numpy.round(obj.rbm_params[3].get_value()[0:3, 0:3], 8)).all(), '6'
		print (c==numpy.round(obj.rbm_params[6].get_value()[0:3, 0:3], 8)).all(), '7'
		print bias==numpy.round(obj.rbm_params[1].get_value()[0],2), '8'

 	elif module == 2:
 		print testing==pretrain_cost, '9'
		print (d==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '10'		
		print (e==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '11'
		print (f==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '12'
		print bias2==numpy.round(obj.params[3].get_value()[0],9), '13'
		print (d==numpy.round(obj.rbm_params[0].get_value()[0:3, 0:3], 8)).all(), '14'
		print (e==numpy.round(obj.rbm_params[3].get_value()[0:3, 0:3], 8)).all(), '15'
		print (f==numpy.round(obj.rbm_params[6].get_value()[0:3, 0:3], 8)).all(), '16'
		print bias3==numpy.round(obj.rbm_params[5].get_value()[0],9), '17'

	elif module == 3:
		print (d==numpy.round(obj.params[0].get_value()[0:3, 0:3], 8)).all(), '18'
		print (e==numpy.round(obj.params[2].get_value()[0:3, 0:3], 8)).all(), '19'
		print (f==numpy.round(obj.params[4].get_value()[0:3, 0:3], 8)).all(), '20'

	elif module == 4:
		print testing==train_cost_error, '21' 


if __name__ == '__main__':
    test_jdy_A()