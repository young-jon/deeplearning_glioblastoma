import matplotlib.pyplot as plt

import cPickle
import gzip
import PIL.Image
import numpy

from jdy_utils import tile_raster_images



'''dataset='/Users/jdy10/Data/mnist/mnist.pkl.gz'
f = gzip.open(dataset, 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()'''
# img=train_set[0][19]
# img.shape
# img.size

# imgplot = plt.imshow(img.reshape(28, 28), cmap="Greys")
# plt.show()

# def show(imgs, n=1):
#     fig = plt.figure()
#     for i in xrange(0, n):
#         fig.add_subplot(1, n, i+1, xticklabels=[], yticklabels=[])
#         if n == 1:
#             img = imgs
#         else:
#             img = imgs[i]
#         plt.imshow(img.reshape(28, 28), cmap="Greys")

def show(imgs, rows=1, columns=1):
    fig = plt.figure()
    for row in [1,0]:
    	for col in xrange(0,columns):
        	fig.add_subplot(row+1, columns, col+1, xticklabels=[], yticklabels=[])
        	if columns == 1:
        		img = imgs
        	if row == 0:
        		img = imgs[col + 6]
        	else:
        		img=imgs[col]
        		#img = imgs[col+(row*6)]
        	plt.imshow(img.reshape(28, 28), cmap="Greys")

#first=show(train_set[0][19:31], rows=2, columns=6)
#sec=show(train_set[0][19:25], n=6)

###SO FAR USE THIS ONE
#need to create an array (like train_set) for my the data I want to plot (should be 100,728). 
#then plot every 20 rows by indexing in the for loop.
'''n_rows = 5
n_columns = 20
image_data = numpy.zeros((29 * n_rows + 1, 29 * n_columns - 1),
                             dtype='uint8')

for idx in xrange(n_rows):
	image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
	                X=train_set[0][19 + (n_columns*idx):39 + (n_columns*idx)],
	                img_shape=(28, 28),
	                tile_shape=(1, n_columns),
	                tile_spacing=(1, 1))

image = PIL.Image.fromarray(image_data)
image.save('samples.png')'''


def create_images(X, n_rows, n_columns):
    image_data = numpy.zeros((29 * n_rows + 1, 29 * n_columns - 1),
                             dtype='uint8')

    for idx in xrange(n_rows):
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
                                            X=X[20*idx:20*(idx+1)],
                                            img_shape=(28, 28),
                                            tile_shape=(1, n_columns),
                                            tile_spacing=(1, 1))

    image = PIL.Image.fromarray(image_data)
    image.save('samples.png')


def create_images_new(X, n_rows, n_columns):
    image_data = numpy.zeros((54 * n_rows + 1, 54 * n_columns - 1),
                             dtype='uint8')

    for idx in xrange(n_rows):
        image_data[54 * idx:54 * idx + 53, :] = tile_raster_images(
                                            X=X[20*idx:20*(idx+1)],
                                            img_shape=(53, 53),
                                            tile_shape=(1, n_columns),
                                            tile_spacing=(1, 1))

    image = PIL.Image.fromarray(image_data)
    image.save('samples.png')






