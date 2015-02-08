import cPickle
import gzip
import os

import numpy

import theano
import theano.tensor as T


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset in quotes.
    :dataset should be saved as a .pkl.gz
    '''

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    #testing for mnist
    print len(train_set[0]) == 50000
    print len(valid_set[0]) == 10000
    print len(test_set[0]) == 10000
    print len(train_set[1]) == 50000
    print len(valid_set[1]) == 10000
    print len(test_set[1]) == 10000

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):

        """ Function that loads the dataset into shared variables. For supervised learning

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets us get around this issue
        return shared_x, T.cast(shared_y, 'int32')
        # can run the code below if don't need labels as integers
        #return shared_x, shared_y 

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

#How to call this function
# datasets = load_data('/Users/jon/Data/mnist/mnist.pkl.gz')
# datasets[0]


def load_data_unsupervised(dataset):
    ''' Loads the unsupervised dataset (there are only x values for train, validation, and test sets). This function is untested.

    :type dataset: string
    :param dataset: the path to the dataset in quotes.
    :dataset should be saved as a .pkl.gz
    '''

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set of type: numpy.ndarray of 2 dimensions 
    #(a matrix) with each row corresponding to an example. 

    def shared_dataset_unsupervised(data_x, borrow=True):

        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x
        
    test_set_x = shared_dataset_unsupervised(test_set)
    valid_set_x = shared_dataset_unsupervised(valid_set)
    train_set_x = shared_dataset_unsupervised(train_set)

    rval = [train_set_x, valid_set_x, test_set_x]
    return rval

def save_short(obj, file_path):
    '''
    short=short-term storage
    Save a class object as a pickle file that has the final state of the shared
    variables (params). See the following theano tutorial url:
    http://deeplearning.net/software/theano/tutorial/loading_and_saving.html

    obj = object to be pickled (e.g. dbn (an instance of the DBN class))
    file_path = string name of new file (full path). If only a file name is given will 
    save in current working directory. e.g. '/Users/jon/models/DBN/model2.pkl' 
    '''
    f = file(file_path, 'wb')
    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def save_med_pkl(obj, file_path):
    '''
    med=medium-term storage
    Save the parameters of an object as a pickle file.

    obj = object with parameters to be pickled. Object must have a .params 
    attribute or this function will not work.
    file_path = string name of new file (full path). file name ends with .pkl
    '''

    assert obj.params, 'save_med_pkl: obj does not have .params attribute'

    f = file(file_path, 'wb')
    for arr in obj.params:
        cPickle.dump(arr.get_value(borrow=True), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def save_med_npy(obj, file_path):
    '''
    med=medium-term storage
    Save the parameters of an object as a numpy file.

    obj = object with parameters to be pickled. Object must have a .params 
    attribute or this function will not work.
    file_path = string name of new file (full path). file name ends with .npy
    
    numpy.save seems to automatically convert a list to a numpy array. seems
    to use dtype=float64.
    '''

    assert obj.params, 'save_med_npy: obj does not have .params attribute'

    out=[]
    for arr in obj.params:
        out.append(arr.get_value())
    numpy.save(file_path, out)




### LOADING SAVED MODELS
### To load dbn object 
    '''
    import cPickle
    f = file('obj.save', 'rb')
    loaded_obj = cPickle.load(f)
    f.close()

    In [57]: loaded_obj.params[0].get_value().shape
    Out[57]: (784, 500)'''

### to load from save_med_pkl
def load_med_pkl(file_path, num_params):
    #'/Users/jon/Models/DBNDA_theano/<file_name>'
    import cPickle
    f = file(file_path, 'rb')
    model = []
    for i in range(num_params):  #use 6 for 3 hidden layers
        model.append(cPickle.load(f))
    f.close()
    return model 

### to load from save_med_npy
    '''
    import numpy
    mod = numpy.load('model_test2.npy')
    '''



### SAVE NOTES
### scikit learn and pylearn2 use pickle, so maybe i should too.

###

'''p=[]
    for arr in dbn.params:
        p.append(arr.get_value())
    numpy.savez('arrays.npz', w1=p[0],b1=p[1],w2=p[2],b2=p[3],w3=p[4],b3=p[5])
'''


'''these functions were taken directly from utils.py in deep learning tutorial code'''

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = numpy.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = numpy.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array

def get_reconstructions(obj, data, batch_size, n_samples=None, reconst_len=None):
    '''calls build_reconstruction_function on any object passed in. Thus, object
    passed in must have a build_reconstruction_function. reshapes and returns 2d
    array. If change version, needs to match up with .build_reconstruction_fn.

    For MNIST train_set_x, creates a list (r) of 5000 numpy.ndarrays. Each of 
    these 5000 arrays is a 10 x 784 array with each row representing a 784-D 
    reconstruction. r is then reshaped and a 2d array is returned with each 
    reconstruction as a row. for MNIST train set, returned array would be 
    50,000 X 784.

    obj = an object with a build_reconstruction_function function
    data = input to create reconstructions from
    n_samples = number of samples of data passed in
    reconst_len = number of elements in each reconstructions

    ### characteristics of r before reshaping
    print type(r)       #list
    print type(r[2])    #numpy.ndarray
    print r[2].shape    #10,784
    print r[2].size     #7840
    print len(r)        #5000
    print r[2][2][0:20]       #[  9.64138480e-01   4.62259240e-01   7.82546112e-01   4.05068000e-01
                              #   5.82822154e-01   6.93803155e-01   8.82764377e-01   8.89362245e-01...
    print r[200][8][0:20]
    print len(r[2][2])  #784
    print type(r[2][2]) #numpy.ndarray'''

    if not n_samples:
        n_samples = data.get_value(borrow=True).shape[0]
    if not reconst_len:
        reconst_len = data.get_value(borrow=True).shape[1]

    n_batches = n_samples / batch_size

    reconstruction_fn = obj.build_reconstruction_function(data=data, batch_size=batch_size)

    ### new versions
    #reconstructions = reconstruction_fn(data.get_value(borrow=True)) ### version 2 works
    reconstructions = reconstruction_fn() ### version 3 works

    assert len(reconstructions[0]) == reconst_len, 'get_reconstructions: wrong len'
    return reconstructions


    ### old version -- works
    # r=[]  #to store reconstructions
    # for batch_index in xrange(n_batches):
    #     reconstructions = reconstruction_fn(index=batch_index)
    #     r.append(reconstructions)

    ### reshape r
    # r_3d_ndarray = numpy.asarray(r)
    # r_2d_ndarray = r_3d_ndarray.reshape(n_samples, reconst_len)

    # assert len(r_2d_ndarray[0]) == reconst_len, 'get_reconstructions: wrong len'

    # return r_2d_ndarray






