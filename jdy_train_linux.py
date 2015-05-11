import numpy
import cPickle
import time
import os
import sys
import gzip
import csv
import random
import pandas as pd
from sklearn import cross_validation
from jdy_Srbm_DAfinetune import run_Srbm_DAfinetune
from jdy_utils import jdy_load_data, shared_dataset_unsupervised
numpy.set_printoptions(threshold=700)


'''SETUP EXPERIMENT HYPERPARAMETERS'''

hidden_layers_sizes = [1200, 400, 100]
pretrain_lr = 0.005
finetune_lr = 0.005
pretraining_epochs = 1
training_epochs = 30   
batch_size = 10
data_path = '/home/jdy10/Data/v2_tcga_data/data_after_fs/data_9476.pkl'  ### this data is not randomized
train_i_path = '/home/jdy10/Data/v2_tcga_data/test_train_splits/85_15/train21_indices_rand_use.pkl'  ### randomized train indices (from original split)
test_i_path = '/home/jdy10/Data/v2_tcga_data/test_train_splits/85_15/test21_indices_rand_use.pkl'   ### randomized test indices
cv_indices_path = '/home/jdy10/Data/v2_tcga_data/test_train_splits/85_15/cv_indices/cv_6400_8.pkl'  ### these indices split the train data
output_folder_path = '/home/jdy10/Output/Srbm_DAfinetune/cv/5_7_15_threelayer_9476/'   ### CHANGE THIS

k = 1  ### don't change this for now
numpy_rng = numpy.random.RandomState(None)  ### change None to 123 for debugging
testing = False  ### don't change unless testing
computer = 'work'  ### doesn't matter unless testing


'''END SETUP'''

hid_layers_list = [[1100,900,400],[1100,900,500]]
first=[1300,1500,1700,1900,2100,2300,2500,2700,2900,3100,3300,3500]
sec=[100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300]
third=[10,50,100,150,200,300,400,500]

for f in first:
	for s in sec:
		if s < f:
			for t in third:
				if t < s:
					hid_layers_list.append([f,s,t])

print hid_layers_list

 # [100,50],
 # [300,50],
 # [300,100],
 # [500,50],
 # [500,100],
 # [500,300],
 # [700,50],
 # [700,100],
 # [700,300],
 # [700,500],
 # [900,50],
 # [900,100],
 # [900,300],
 # [900,500],
 # [900,700],
 # [1100,50],
 # [1100,100],
 # [1100,300],
 # [1100,500],
 # [1100,700],
 # [1100,900],
 # [1300,50],
 # [1300,100],
 # [1300,300],
 # [1300,500],
 # [1300,700],
 # [1300,900],
 # [1300,1100],
 # [1500,50],
 # [1500,100],
 # [1500,300],
 # [1500,500],
 # [1500,700],
 # [1500,900],
 # [1500,1100],
 # [1500,1300],
 # [1700,50],
 # [1700,100],
 # [1700,300],
 # [1700,500],
 # [1700,700],
 # [1700,900],
 # [1700,1100],
 # [1700,1300],
 # [1700,1500],
 # [1900,50],
 # [1900,100],
 # [1900,300],
 # [1900,500],
 # [1900,700],
 # [1900,900],
 # [1900,1100],
 # [1900,1300],
 # [1900,1500],
 # [1900,1700],
 # [2100,50],
 # [2100,100],
 # [2100,300],
 # [2100,500],
 # [2100,700],
 # [2100,900],
 # [2100,1100],
 # [2100,1300],
 # [2100,1500],
 # [2100,1700],
 # [2100,1900],
 # [2300,50],
 # [2300,100],
 # [2300,300],
 # [2300,500],
 # [2300,700],
 # [2300,900],
 # [2300,1100],
 # [2300,1300],
 # [2300,1500],
 # [2300,1700],
 # [2300,1900],
 # [2300,2100],
 # [2500,50],
 # [2500,100],
 # [2500,200],
 # [2500,300],
 # [2500,500],
 # [2500,700],
 # [2500,900],
 # [2500,1100],
 # [2500,1300],
 # [2500,1500],
 # [2500,1700],
 # [2500,1900],
 # [2500,2100],
 # [2500,2300],
 # [2700,50],
 # [2700,100],
 # [2700,300],
 # [2700,500],
 # [2700,700],
 # [2700,900],
 # [2700,1100],
 # [2700,1300],
 # [2700,1500],
 # [2700,1700],
 # [2700,1900],
 # [2700,2100],
 # [2700,2300],
 # [2700,2500],
 # [2900,50],
 # [2900,100],
 # [2900,300],
 # [2900,500],
 # [2900,700],
 # [2900,900],
 # [2900,1100],
 # [2900,1300],
 # [2900,1500],
 # [2900,1700],
 # [2900,1900],
 # [2900,2100],
 # [2900,2300],
 # [2900,2500],
 # [2900,2700],
 # [3100,50],
 # [3100,100],
 # [3100,300],
 # [3100,500],
 # [3100,700],
 # [3100,900],
 # [3100,1100],
 # [3100,1300],
 # [3100,1500],
 # [3100,1700],
 # [3100,1900],
 # [3100,2100],
 # [3100,2300],
 # [3100,2500],
 # [3100,2700],
 # [3100,2900],
 # [3300,50],
 # [3300,100],
 # [3300,300],
 # [3300,500],
 # [3300,700],
 # [3300,900],
 # [3300,1100],
 # [3300,1300],
 # [3300,1500],
 # [3300,1700],
 # [3300,1900],
 # [3300,2100],
 # [3300,2300],
 # [3300,2500],
 # [3300,2700],
 # [3300,2900],
 # [3300,3100],
 # [3500,50],
 # [3500,100],
 # [3500,300],
 # [3500,500],
 # [3500,700],
 # [3500,900],
 # [3500,1100],
 # [3500,1300],
 # [3500,1500],
 # [3500,1700],
 # [3500,1900],
 # [3500,2100],
 # [3500,2300],
 # [3500,2500],
 # [3500,2700],
 # [3500,2900],
 # [3500,3100],
 # [3500,3300]
 # ]
 # [3700],
 # [3900],
 # [4100],
 # [4300],
 # [4500],
 # [4700],
 # [4900],
 # [5100],
 # [5300],
 # [5500],
 # [5700],
 # [5900],
 # [6100],
 # [6300],
 # [6500],
 # [6700],
 # [6900],
 # [7100],
 # [7300],
 # [7500],
 # [7700],
 # [7900],
 # [8100],
 # [8300],
 # [8500],
 # [8700],
 # [8900],
 # [9100],
 # [9300],
 # [9500]
 # ]

# hid_layers_list = [
#  [400, 200, 20],
#  [400, 200, 100],
#  [400, 200, 180],
#  [1400, 200, 20],
#  [1400, 200, 100],
#  [1400, 200, 180],
#  [1400, 600, 20],
#  [1400, 600, 100],
#  [1400, 600, 180],
#  [1400, 1000, 20],
#  [1400, 1000, 100],
#  [1400, 1000, 180]]
 
# hid_layers_list = [
#  [2400, 200, 20],
#  [2400, 200, 100],
#  [2400, 200, 180],
#  [2400, 600, 20],
#  [2400, 600, 100],
#  [2400, 600, 180],
#  [2400, 1000, 20],
#  [2400, 1000, 100],
#  [2400, 1000, 180]]



if not testing:
	### new
	for h_index, h in enumerate(hid_layers_list):
		hidden_layers_sizes = h
		pretrain_lr = 0.003025607848026375
		finetune_lr = 0.003260677959656419
		pretraining_epochs = 14
		batch_size = 10


		# choice = random.choice([1,2])
		# if choice == 1:
		# 	pretrain_lr = 0.00011502592279277739
		# 	finetune_lr = 0.011003312912625188
		# 	pretraining_epochs = 12
		# 	batch_size = 20
		# elif choice == 2:
		# 	pretrain_lr = 0.003025607848026375
		# 	finetune_lr = 0.003260677959656419
		# 	pretraining_epochs = 14
		# 	batch_size = 10

		# num_layers = random.choice([2,3,4,5,6,7])
		# hidden_layers_sizes = [500] * num_layers
		# hidden_layers_sizes[0] = random.randrange(500,2501)
		# hidden_layers_sizes[1] = random.randrange(200,2001)
		# if num_layers > 2:
		# 	hidden_layers_sizes[2] = random.randrange(20,1501)
		# if num_layers > 3:
		# 	for i in range(3, num_layers):
		# 		hidden_layers_sizes[i] = random.randrange(10, 1001)
		# print hidden_layers_sizes

		# num_layers = random.choice([2,3,4,5,6,7])
		# hidden_layers_sizes = [500] * num_layers
		# hidden_layers_sizes[0] = random.randrange(500,2501)
		# hidden_layers_sizes[1] = random.randrange(100,hidden_layers_sizes[0] - 100)
		# if num_layers > 2:
		# 	for i in range(2, num_layers):
		# 		hidden_layers_sizes[i] = random.randrange(10, hidden_layers_sizes[i-1] + 1)
		# print hidden_layers_sizes

		# batch_size = random.choice([5,10,20,40,50,100])
		# pretraining_epochs = random.choice([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
		# pretrain_lr = 10** -(random.uniform(1.5, 5.5))
		# finetune_lr = 10** -(random.uniform(1.5, 5.5))
		### end new

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
		for i, (train_cv_indices, valid_cv_indices) in enumerate(cv_indices[0:4]):   
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


		### new code for saving all errors from all hyperparams sets in one file
		if h_index == 0:
			name_test = str(dim) + 'all_test_squared_error'
			file_path_test = output_folder_path + name_test + time.strftime("%m%d%Y_%H;%M") + '.csv'
			name_train = str(dim) + 'all_train_squared_error'
			file_path_train = output_folder_path + name_train + time.strftime("%m%d%Y_%H;%M") + '.csv'
			name_cost = str(dim) + 'all_train_cost'
			file_path_cost = output_folder_path + name_cost + time.strftime("%m%d%Y_%H;%M") + '.csv'

		with open(file_path_test, 'a') as f_test:
			row_to_disk = list(test_mse_avg)
			row_to_disk.insert(0, str(srbm['params']))
			writer = csv.writer(f_test)
			writer.writerow(row_to_disk)

		with open(file_path_train, 'a') as f_train:
			row_to_disk = list(train_mse_avg)
			row_to_disk.insert(0, str(srbm['params']))
			writer = csv.writer(f_train)
			writer.writerow(row_to_disk)

		with open(file_path_cost, 'a') as f_cost:
			row_to_disk = list(train_cost_avg)
			row_to_disk.insert(0, str(srbm['params']))
			writer = csv.writer(f_cost)
			writer.writerow(row_to_disk)







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



