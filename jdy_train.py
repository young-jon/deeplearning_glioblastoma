import cPickle
from sklearn import cross_validation
from jdy_Srbm_DAfinetune import run_Srbm_DAfinetune
from jdy_utils import jdy_load_data, shared_dataset_unsupervised



hidden_layers_sizes = [1000, 500, 250, 30]
pretrain_lr = 0.6
finetune_lr = 0.6
pretraining_epochs = 1 
training_epochs = 5
k = 1
batch_size = 10
computer = 'work'
data_path = '/Users/jdy10/Data/data_after_fs/data_2784.pkl'
cv_indices_path = '/Users/jdy10/Data/cv_indices/cv_6108_6.pkl'





### get test set and train set as two numpy.array
train, test = jdy_load_data(data_path)

### get indices to split train set into train_cv and valid_cv
f = open(cv_indices_path, 'rb') 
cv_indices = cPickle.load(f)
f.close()


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
								k=k, batch_size=batch_size, computer='work')
	
