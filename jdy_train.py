import numpy
import cPickle
import time
import os
import sys
import gzip
from sklearn import cross_validation
from jdy_Srbm_DAfinetune import run_Srbm_DAfinetune
from jdy_utils import jdy_load_data, shared_dataset_unsupervised


'''SETUP EXPERIMENT HYPERPARAMETERS'''

hidden_layers_sizes = [1000, 500, 250, 30]
pretrain_lr = 0.1
finetune_lr = 0.1
pretraining_epochs = 1
training_epochs = 5
batch_size = 10
computer = 'work'
data_path = '/Users/jdy10/Data/data_after_fs/data_2784.pkl'
cv_indices_path = '/Users/jdy10/Data/cv_indices/cv_6108_6.pkl'
k = 1  ### don't change this for now
numpy_rng = numpy.random.RandomState(None)  ### change None to 123 for debugging

'''END SETUP'''

###TESTING
f = gzip.open('/Users/jdy10/Data/mnist/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_set_x = shared_dataset_unsupervised(train_set[0])
test_set_x = shared_dataset_unsupervised(test_set[0])
srbm = run_Srbm_DAfinetune(train_set_x=train_set_x, test_set_x=test_set_x, 
							hidden_layers_sizes=[1000, 500, 250, 30],
							pretrain_lr=0.1, finetune_lr=0.1,
							pretraining_epochs=1,
							training_epochs=5, k=1, 
							batch_size=10, computer='work', 
							numpy_rng=numpy.random.RandomState(123))



'''
### get test set and train set as two numpy.array
train, test = jdy_load_data(data_path)

### get indices to split train set into train_cv and valid_cv
f = open(cv_indices_path, 'rb') 
cv_indices = cPickle.load(f)
f.close()


### run run_Srbm_DAfinetune for each k fold
start_time = time.time()
for i, (train_cv_indices, valid_cv_indices) in enumerate(cv_indices):
	print i, train_cv_indices, valid_cv_indices
	train_set_x = shared_dataset_unsupervised(train[train_cv_indices])
	test_set_x = shared_dataset_unsupervised(train[valid_cv_indices])
	print i, train_set_x, test_set_x

	srbm = run_Srbm_DAfinetune(train_set_x=train_set_x, 
								test_set_x=test_set_x, 
								hidden_layers_sizes=hidden_layers_sizes,
								pretrain_lr=pretrain_lr, 
								finetune_lr=finetune_lr,
								pretraining_epochs=pretraining_epochs,
								training_epochs=training_epochs,
								k=k, batch_size=batch_size, computer='work',
								numpy_rng=numpy_rng)

	if i == 0:
		end_time1 = time.time()
		print 'first fold ran for ', (end_time1 - start_time)/60.


end_time = time.time()
print >> sys.stderr, ('The cross-validation code for file ' +
	                      os.path.split(__file__)[1] +
	                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
'''
	
