import os
import numpy as np 
import cv2
import utils as ut
import time

class Caltech101:
    """ Caltech101 dataset
    """
    def __init__(self, root, resize_h = 300, resize_w = 300):
        self.root = ut.test_postfix_dir(root)
        # assert(os.path.exists(self.root), "")
        self.clsses = os.listdir(self.root)
        self.clss2idx = dict([(clss, i) for i, clss in enumerate(self.clsses)])

        self.img_names = []
        self.num_clss = len(self.clsses)
        self.size = 0
        self.resize_h = 300
        self.resize_w = 300
        # load name of every picture in each class
        self.each_clss_size = np.zeros((self.num_clss,), dtype = int)
        for i, clss in enumerate(self.clsses):
            pics = os.listdir(self.root + clss)
            self.img_names.append(pics)
            self.each_clss_size[i] = len(pics)
            self.size += len(pics)

    def __len__(self):
        return self.size

    def load_image(self, image_path):
        image = cv2.imread(image_path, flags = cv2.IMREAD_GRAYSCALE)
        # image = imread(image_path, as_gray=True)
        return self.preprocess(image)

    def preprocess(self, image):
        image = cv2.resize(image, (self.resize_w, self.resize_h))
        # normlize to [0,1]
        # image = image / 255.0
        return image

    def xy(self, using_clss = -1, num_clss_train_image = 30, num_clss_test_image = 50,
            shuffle_images_in_class = True, shuffle_indices = True):
        """ Load dataset and do train-test-split.

        :params:
            using_clss: number of total classes used in training. 
            num_clss_train_image: number of images in each class used in training. Better less than 30, for 
                some class would not have much left for test images.
            num_clss_test_image: number of images in each class used in testing. The actual number would be 
                less, for some classes have less images than expected. 
            shuffle_images_in_class: shuffle the selection of images in each class.
            shuffle_indices: shuffle the formed sets.
        """
        if using_clss <= 0 or using_clss >= self.num_clss:
            using_clss = self.num_clss

        height = self.resize_h
        width = self.resize_w
        Xtrain = []
        ytrain = []
        Xtest = []
        ytest = []

        for clss, clss_images in enumerate(self.img_names[:using_clss]):
            xtemp = np.zeros((num_clss_train_image, height, width), dtype = np.uint8)
            ytemp = np.zeros((num_clss_train_image, ))

            testlen = min(len(clss_images), num_clss_test_image)
            xtesttemp = np.zeros((testlen, height, width), dtype = np.uint8)
            ytesttemp = np.zeros((testlen, ))

            if shuffle_images_in_class:
                clss_images = np.array(clss_images, dtype = str)
                np.random.shuffle(clss_images)

            clss_images = clss_images[: num_clss_test_image + num_clss_train_image]
            for i, image in enumerate(clss_images):
                image_path = self.root + self.clsses[clss] + os.sep + image
                image = self.load_image(image_path)

                if i < num_clss_train_image: # only 30 image using in training 
                    xtemp[i, :] = image
                    ytemp[i] = clss
                elif i < num_clss_train_image + num_clss_test_image: # left for testing
                    xtesttemp[i - num_clss_train_image, :] = image
                    ytesttemp[i - num_clss_train_image] = clss

            Xtrain.append(xtemp)
            ytrain.append(ytemp)
            Xtest.append(xtesttemp)
            ytest.append(ytesttemp)
            print("Loading clss \"{}\" images done.".format(self.clsses[clss]))

        Xtrain = np.concatenate(Xtrain, axis = 0)
        Xtest = np.concatenate(Xtest, axis = 0)
        ytrain = np.concatenate(ytrain, axis = 0)
        ytest = np.concatenate(ytest, axis = 0)

        if shuffle_indices: 
            indices = np.arange(0,Xtrain.shape[0])
            np.random.shuffle(indices)
            Xtrain = Xtrain[indices]
            ytrain = ytrain[indices]

            indices = np.arange(0,Xtest.shape[0])
            np.random.shuffle(indices)
            Xtest = Xtest[indices]
            ytest = ytest[indices]   

        return Xtrain, ytrain, Xtest, ytest

    def idx2name(self, target):
        target = np.array(target)
        labels = np.full_like(target, 0, dtype=str)
        for i, ele in enumerate(target):
            labels[i] = self.clsses[ele]

        return labels

    def name2idx(self, name):
        name = np.array(name)
        labels = np.full_like(name, 0, dtype=int)
        for i, ele in enumerate(name):
            labels[i] = self.clss2idx[ele]
        return labels

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    startime = time.clock()
    root = "~/"
    dataset = Caltech101(root)
    X, y, X2, y2 = dataset.xy(using_clss=5)
    print(X.shape)
    print(y.shape)
    print(X2.shape)
    print(y2.shape)
    print(y)
    endtime = time.clock()
    print("Using {} s".format(endtime - startime))

    plt.imshow(X[0])
    plt.show()