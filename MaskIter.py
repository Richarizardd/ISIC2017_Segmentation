import numpy as np
import sys, os
import random
import cv2
import time
import helpers
import mxnet as mx
from mxnet.io import DataIter
from mxnet.io import DataBatch
from keras.preprocessing.image import *
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

np.random.seed(1301)
random.seed(1301)

class MaskIter(DataIter):
    def __init__(self, 
                 root_dir, 
                 flist_name,
                 data_name="data",
                 label_name="softmax_label",
                 batch_size=16,
                 augment=True,
                 shuffle=False):

        self.root_dir = root_dir
        self.flist_name = self.root_dir+flist_name
        self.data_name, self.label_name = data_name, label_name
        self.augment = augment
        self.mean = [177.11988924, 137.8878728, 123.15429299]
  
        super(MaskIter, self).__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_lines = []
        self.epoch = 0
        self.label_files = []
        self.image_files = []

        self.curr_batch = 1

        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.cursor = -1
        self.read_lines()
        self.data, self.label = self._read()
        self.reset()


    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data, label = {}, {}

        data_batch = []
        label_batch = []
        for i in range(0, self.batch_size):
            line = self.get_line()
            img_name = line.rstrip('\n')+".jpg"
            label_name =  img_name[:-4]+"_segmentation.png"
            curr_data, curr_label = self._read_img(img_name, label_name)
            data_batch.append(curr_data)
            label_batch.append(curr_label)

        data_batch = np.vstack(data_batch)
        label_batch = np.vstack(label_batch)
        data[self.data_name] = data_batch
        label[self.label_name] = label_batch

        return list(data.items()), list(label.items())

    def _read_img(self, img_name, label_name):
        img_path = os.path.join(self.root_dir, "JPEGImages", img_name)
        label_path = os.path.join(self.root_dir, "SegmentationClass", label_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype(float)  # Image.open(img_path).convert("L")
        label = cv2.imread(label_path).astype(float)  # Image.open(label_path).convert("L")

        self.image_files.append(img_path)
        self.label_files.append(label_path)

        if self.augment:
            img, label = self.random_transform(img.copy(), label.copy())

        img[:,:,0] -= self.mean[0]
        img[:,:,1] -= self.mean[1]
        img[:,:,2] -= self.mean[2]

        img /= (np.std(img, axis=2, keepdims=True) + 1e-7)
        label = label[:,:,0]
        label /= 256.

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)  # (c, h, w)
        img = np.expand_dims(img, axis=0)  # (1, c, h, w)
        label = np.array(label)  # (h, w)
        label = label.reshape(1, label.shape[0] * label.shape[1])

        return img, label

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.data]
        # print "data : " + str(res)
        return res

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        res = [(k, tuple(list(v.shape[0:]))) for k, v in self.label]
        print "label : " + str(res)
        return res

    def reset(self):
        self.cursor = -1
        self.read_lines()
        self.label_files = []
        self.image_files = []
        self.epoch += 1
        self.curr_batch = 1
        print "Epoch:", self.epoch

    def getpad(self):
        return 0

    def read_lines(self):
        self.current_line_no = -1;
        with open(self.flist_name, 'r') as f:
            self.file_lines = f.readlines()
            if self.shuffle:
                random.shuffle(self.file_lines)

    def get_line(self):
        self.current_line_no += 1
        return self.file_lines[self.current_line_no]


    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def eof(self):
        res = self.cursor >= self.num_data
        return res

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            print "Batch:", str(self.curr_batch*self.batch_size)+"/"+str(len(self.file_lines))
            self.curr_batch += 1
            self.data, self.label = self._read()
            res = DataBatch(data=[mx.nd.array(self.data[0][1])], label=[mx.nd.array(self.label[0][1])], pad=self.getpad(), index=None)
            return res
        else:
            raise StopIteration

    def elastic_deformation(self, img, label, alpha=256*0.2, sigma=256*0.08, alpha_affine=256*0.08, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
             Convolutional Neural Networks applied to Visual Document Analysis", in
             Proc. of the International Conference on Document Analysis and
             Recognition, 2003.

         Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(1301)

        shape = img.shape
        shape_size = shape[:2]
        
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, shape_size[::-1])
        label = cv2.warpAffine(label, M, shape_size[::-1])
        
        # random elastic deformation
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dz = np.zeros_like(dx)
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        img = map_coordinates(img, indices, order=1, mode='reflect').reshape(shape)
        label = map_coordinates(label, indices, order=1, mode='reflect').reshape(shape)
        return img, label

    def random_transform(self, x, y, 
        rotation_range=180, shear_range=0, zoom_range=[1,1], zoom_maintain_shape=True, 
        elastic=True, channel_shift_range=20, horizontal_flip=True, vertical_flip=True,
        img_row_index=0, img_col_index=1, img_channel_index=2):

        # use composition of homographies to generate final transform that
        # needs to be applied
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        tx = 0
        ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
        if zoom_maintain_shape:
            zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        
        x = apply_transform(x, transform_matrix, img_channel_index, fill_mode='constant', cval=0)
        y = apply_transform(y, transform_matrix, img_channel_index, fill_mode='constant', cval=0)

        if elastic:
            if random.randint(0, 100) < 25:
                x, y = self.elastic_deformation(x, y)
        
        if channel_shift_range != 0:
            x = random_channel_shift(x, channel_shift_range, img_channel_index)

        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)
        return x, y