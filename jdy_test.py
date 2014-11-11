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


if __name__ == '__main__':
    test_jdy_A()