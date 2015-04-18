import numpy
import cPickle
import time
import os
import sys
import gzip
import csv
import pandas as pd
from sklearn import cross_validation
from jdy_Srbm_DAfinetune import run_Srbm_DAfinetune
from jdy_utils import jdy_load_data, shared_dataset_unsupervised
numpy.set_printoptions(threshold=700)


'''SETUP EXPERIMENT HYPERPARAMETERS'''

hidden_layers_sizes = [1000]
pretrain_lr = 0.009
finetune_lr = 0.005
pretraining_epochs = 1
training_epochs = 10
batch_size = 5
data_path = '/Users/jdy10/Data/v2_tcga_data/data_after_fs/data_9476.pkl'  ### this data is not randomized
train_i_path = '/Users/jdy10/Data/v2_tcga_data/test_train_splits/85_15/train21_indices_rand_use.pkl'  ### randomized train indices (from original split)
test_i_path = '/Users/jdy10/Data/v2_tcga_data/test_train_splits/85_15/test21_indices_rand_use.pkl'   ### randomized test indices
cv_indices_path = '/Users/jdy10/Data/v2_tcga_data/test_train_splits/85_15/cv_indices/cv_6400_8.pkl'  ### these indices split the train data
output_folder_path = '/Users/jdy10/Output/Srbm_DAfinetune/cv/'

k = 1  ### don't change this for now
numpy_rng = numpy.random.RandomState(None)  ### change None to 123 for debugging
testing = False  ### don't change unless testing
computer = 'work'  ### doesn't matter unless testing


'''END SETUP'''



if not testing:
	# for h in [[10], [100], [500], [900], [1500], [3000], [6000], [9000], [9856]]:
	# for h in [[200],[300],[400],[600],[700],[800],[1000],[1100],[1200],[1300],[1400],[1600],[1700],[1800],[1900],[2000],[2100],[2200],[2300],[2400],[2500],[2600],[2700],[2800],[2900]]:
	# hidden_layers_sizes = h
	### initialize containers for writing to file
	test_mse=[]; train_mse=[]; train_cost=[]
	### get test set and train set as two numpy.array
	train, test = jdy_load_data(data_path=data_path, train_i_path=train_i_path, test_i_path=test_i_path, shared=False)
	### doesn't set shared=True here because still need to do a cv train/valid split below, which is more difficult to do if i created theano shared vars.
	dim = train.shape[1]

	### get indices to split train set into train_cv and valid_cv
	f = open(cv_indices_path, 'rb') 
	cv_indices = cPickle.load(f)
	f.close()


	### run run_Srbm_DAfinetune for each k fold
	start_time = time.time()
	for i, (train_cv_indices, valid_cv_indices) in enumerate(cv_indices[0:3]):   ### change this indexing to do all
		print 'IDX:', i, train_cv_indices, valid_cv_indices
		train_set_x = shared_dataset_unsupervised(train[train_cv_indices])
		test_set_x = shared_dataset_unsupervised(train[valid_cv_indices])
		print ( 'CV:' + str(i) + ' train:' + str(train_set_x.get_value().shape)+ 
							' test:' + str(test_set_x.get_value().shape) )

		srbm = run_Srbm_DAfinetune(train_set_x=train_set_x, 
									test_set_x=test_set_x, 
									hidden_layers_sizes=hidden_layers_sizes,
									pretrain_lr=pretrain_lr, 
									finetune_lr=finetune_lr,
									pretraining_epochs=pretraining_epochs,
									training_epochs=training_epochs,
									k=k, batch_size=batch_size, computer=computer,
									numpy_rng=numpy_rng, testing=testing)

		if i == 0:
			end_time1 = time.time()
			print 'first fold ran for ', (end_time1 - start_time)/60.

		test_mse.append(srbm['test_mse'])
		train_mse.append(srbm['train_mse'])
		train_cost.append(srbm['train_cost'])

		### write test_mse to file
		if i == 0:
			name = 'test_mse_' + str(dim) + str(srbm['params']).replace(' ','').replace("'","")
			file_path = output_folder_path + name + time.strftime("%m%d%Y_%H;%M") + '.csv'

		with open(file_path, 'a') as f:
			writer=csv.writer(f)
			writer.writerow(srbm['test_mse'])


	end_time = time.time()
	print >> sys.stderr, ('The code for file ' +
		                      os.path.split(__file__)[1] +
		                      ' ran for %.2fm' % ((end_time-start_time) / 60.))


	### for collecting errors
	print test_mse
	print train_mse
	print train_cost
	test_mse_avg = numpy.asarray(test_mse).mean(axis=0)
	train_mse_avg = numpy.asarray(train_mse).mean(axis=0)
	train_cost_avg = numpy.asarray(train_cost).mean(axis=0)
	print test_mse_avg
	print train_mse_avg
	print train_cost_avg

	df_to_disk = pd.DataFrame([test_mse_avg, train_mse_avg, train_cost_avg],
								index=[srbm['params'], '', ''])
	df_to_disk['error_type'] = ['test_mse', 'train_mse', 'train_cost']

	### create file name and save as .csv
	name = str(dim) + str(srbm['params']).replace(' ','').replace("'","")
	file_path = output_folder_path + name + time.strftime("%m%d%Y_%H;%M") + '.csv'
	df_to_disk.to_csv(file_path)






###TESTING ONLY.  Users do not change this code.
if testing:
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
								numpy_rng=numpy.random.RandomState(123), 
								testing=testing)



