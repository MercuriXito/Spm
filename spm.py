from collections import Counter
import numpy as np 
import cv2
import time, json
from tqdm import tqdm
from utils import test_and_make_dir, test_postfix_dir, currentTime, json_dump

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class SpatialPyramidClassifier:
    """ 
    :params:
        M: size of vocabulary, number of clusters' centers.
        L: number of levels.
        cluster_batch: batch of samples used in training clusters
        do_train_clusters: train cluster or not. (if not, need `load_model` first)
        do_train_classifier: train classifier or not. (if not, need `load_model` first)
        max_num_sample_train_cluster: maximum number of samples used in training clusters, 
            since it only use the subset of trainset. 
        save_root: save folder
        use_matrix_in_kernel: when doing matching in kernel function, using Matrix Way or not. 
            Note that Matrix way **consume large memory**.
    """
    def __init__(self, M = 200, L = 2, cluster_batch = 1000, 
        do_train_clusters = True, do_train_classifier = True,
        max_num_sample_train_cluster = 20000, save_root = "save", 
        use_matrix_in_kernel = False, save_sift_feature = True):

        self.saveroot = test_postfix_dir(save_root)
        test_and_make_dir(self.saveroot)

        self.do_train_clusters = do_train_clusters
        self.do_train_classifier = do_train_classifier
        self.save_sift_feature = save_sift_feature

        self.use_matrix_in_kernel = use_matrix_in_kernel

        self.L = L 
        self.M = M
        self.num_cells = (4 **(L + 1) - 1) // 3
        self.sp_dim = M * self.num_cells # dim of spatial-pyramid-feature

        # self.clusters = KMeans(M) # clusters as vocabulary
        self.clusters_batch = cluster_batch
        self.clusters = MiniBatchKMeans(M, batch_size=self.clusters_batch)
        # self.classifier = SVC(kernel="precomputed")
        self.classifier = SVC(kernel=self.spatial_pyramid_matching_kernel)
        self.MAX_NUM_SAMPLE_TRAIN_CLUSTER  = max_num_sample_train_cluster # maximum number of training samples in training KMeans


    def save_model(self):
        joblib.dump(self.clusters, self.saveroot + "clusters_" + currentTime() + ".pt")
        joblib.dump(self.classifier, self.saveroot + "classifier_" + currentTime() + ".pt")

    def load_model(self, cluster_file = None, classifier_file= None):
        if cluster_file:
            self.clusters =  joblib.load(cluster_file)
        if classifier_file:
            self.classifier = joblib.load(classifier_file)

    def save_feature(self, feature):
        np.save(self.saveroot + "feature_" + currentTime() + ".npy", feature)

    def load_feature(self, filename):
        return np.load(filename)

    def _set_feature_hw(self, images):
        h, w = images.shape[1:]
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

    def feature_dense_sift_one_image(self, image):
        """ Extract dense-sift features from one image
        """
        sift = cv2.xfeatures2d.SIFT_create()
        h, w = image.shape
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

        kp = [cv2.KeyPoint(x,y, patch_size[0]) for y in coord_y
                                        for x in coord_x]
        _, dense_sift = sift.compute(image, kp)
        return dense_sift

    def feature_dense_sift(self, images):
        """ Extract Dense sift features from a batch of images
        """
        sift = cv2.xfeatures2d.SIFT_create()
        h, w = images.shape[1:]
        patch_size = (16, 16)
        step_size = 8

        coord_y = [y for y in range(step_size, h, step_size)]
        coord_x = [x for x in range(step_size, w, step_size)]
        self.num_feature_h = len(coord_y)
        self.num_feature_w = len(coord_x)

        kp = [cv2.KeyPoint(x,y, patch_size[0]) for y in coord_y
                                        for x in coord_x]
        features = np.zeros([images.shape[0], self.num_feature_h * self.num_feature_w, 128])
        for i, image in enumerate(tqdm(images)):
            _, dense_sift = sift.compute(image, kp)
            features[i] = dense_sift

        return features

    def train_clusters(self, features):
        """ use random subset of patch features to train KMeans as dictionary.
        """
        # sample from features
        num_samples, num_points = features.shape[:2]
        size_train_set = min( num_samples * num_points , self.MAX_NUM_SAMPLE_TRAIN_CLUSTER)
        indices = np.random.choice(num_samples * num_points , size_train_set)

        # todo: ref might consume large additional memory
        trainset = features.reshape(num_points * num_samples, -1)[indices, :]

        # train and predict
        # self.clusters.fit(trainset)
        print("Training MiniBatch KMeans")
        for i in tqdm(range(size_train_set // self.clusters_batch + 1)):
            start_idx = self.clusters_batch * i
            end_idx = min(self.clusters_batch * (i + 1), size_train_set)
            if end_idx - start_idx == 0:
                break
            batch = trainset[start_idx:end_idx, :]
            self.clusters.partial_fit(batch)

    def toBOF(self, features):
        """ convert lower feature to bags of feature
        """
        num_samples, num_points = features.shape[:2]
        vocab = self.clusters.predict(features.reshape(num_samples * num_points, -1))
        return vocab.reshape(num_samples, num_points, )

    def spatial_pyramid_matching_precomputed(self, x):
        """ precomputed function of svm, calculate the matching score of two vector
        """
        num_samples, num_features = x.shape
        x = x.reshape(-1, self.num_feature_h, self.num_feature_w)
        x_feature = np.zeros((num_samples, self.num_cells, self.M))

        # extract feature
        for level in range(0, self.L + 1):
            if level == 0:
                coef = 1 / (2 ** self.L)
            else:
                coef = 1 / (2 ** (self.L + 1 - level))

            num_level_cells = 4 ** level
            num_segments = 2 ** level
            cell_hs = np.linspace(0, self.num_feature_h, 2 ** level + 1, dtype=int)
            cell_ws = np.linspace(0, self.num_feature_w, 2 ** level + 1, dtype=int)

            # cells in each level
            for cell in range(num_level_cells):

                idx_y = cell // num_segments
                idx_x = cell % num_segments

                # histogram of BOF in one cell
                x_block_feature = x[:, cell_hs[idx_y] :cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples):
                    counter = Counter(x_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M))
                    level_feature = np.zeros((self.M, )) # 注意可能在cell内有未出现的 类
                    level_feature[counts[:,0]] = counts[:,1]
                    x_feature[s, count, :] = level_feature * coef
            
        x_feature = x_feature.reshape(num_samples, -1)
        xf1 = x_feature.reshape(num_samples, 1, -1).repeat(num_samples, axis = 1)
        xf2 = np.transpose(xf1, (1, 0, 2))

        return np.min([xf1, xf2], axis = 0).sum(axis = -1)

    def spatial_pyramid_matching_kernel(self, x, y):
        """ spatial pyramid matching kernel function of svm, 
            calculate the matching score of two vector
        """
        num_samples_x, num_features = x.shape
        num_samples_y, num_features = y.shape

        x = x.reshape(-1, self.num_feature_h, self.num_feature_w)
        y = y.reshape(-1, self.num_feature_h, self.num_feature_w)

        x_feature = np.zeros((num_samples_x, self.num_cells, self.M))
        y_feature = np.zeros((num_samples_y, self.num_cells, self.M))

        # extract feature
        count = 0
        for level in range(0, self.L + 1):
            if level == 0:
                coef = 1 / (2 ** self.L)
            else:
                coef = 1 / (2 ** (self.L + 1 - level))

            num_level_cells = 4 ** level
            num_segments = 2 ** level
            cell_hs = np.linspace(0, self.num_feature_h, 2 ** level + 1, dtype=int)
            cell_ws = np.linspace(0, self.num_feature_w, 2 ** level + 1, dtype=int)

            # cells in each level
            for cell in range(num_level_cells):

                idx_y = cell // num_segments
                idx_x = cell % num_segments

                # histogram of BOF in one cell
                x_block_feature = x[:, cell_hs[idx_y] :cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples_x):
                    counter = Counter(x_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M), dtype = int)
                    level_feature = np.zeros((self.M, )) # 注意可能在cell内有未出现的类
                    if counts.size != 0:
                        level_feature[counts[:,0]] = counts[:,1]
                    x_feature[s, count, :] = level_feature * coef

                y_block_feature = y[:, cell_hs[idx_y] :cell_hs[idx_y + 1], cell_ws[idx_x]:cell_ws[idx_x + 1]]
                for s in range(num_samples_y):
                    counter = Counter(y_block_feature[s].flatten())
                    counts = np.array(counter.most_common(self.M), dtype = int)
                    level_feature = np.zeros((self.M, ))
                    if counts.size != 0:
                        level_feature[counts[:, 0]] = counts[:, 1]
                    y_feature[s, count, :] = level_feature * coef
            
                count += 1
            
        x_feature = x_feature.reshape(num_samples_x, -1)
        y_feature = y_feature.reshape(num_samples_y, -1)

        # 此处直接用matrix计算的话两个repeat的内存占用会非常大，仅能对非常小的样本使用
        # 如果改成循环内存占用会小很多，但是牺牲时间效率
        if self.use_matrix_in_kernel:
            xf = x_feature.reshape(num_samples_x, 1, -1).repeat(num_samples_y, axis = 1)

            yf = y_feature.reshape(num_samples_y, 1, -1).repeat(num_samples_x, axis = 1)
            yf = np.transpose(yf, (1, 0, 2))
            t = np.min([xf, yf], axis = 0).sum(axis = -1)
        else:
            t = np.zeros((num_samples_x, num_samples_y))
            for i in range(num_samples_x):
                for j in range(num_samples_y):
                    a = x_feature[i,:]
                    b = y_feature[j,:]
                    t[i][j] = np.min([a, b], axis = 0).sum(axis = -1)

        return t

    def train_classifier(self, X_train, y_train):

        print(X_train.shape)
        self.classifier.fit(X_train, y_train)
        y_predict = self.classifier.predict(X_train)

        report = classification_report(y_train, y_predict)
        print("Classifier Training Report: \n {}".format(report))


    def predict_clss(self, X):
        return self.classifier.predict(X)

    def train(self, images, labels, precompute_feature = None):

        if precompute_feature is None:
            print("Extracting Dense sift feature")
            low_features = self.feature_dense_sift(images)
            if self.save_sift_feature:
                self.save_feature(low_features)
        else:
            print("Using pre-computed feature")
            low_features = precompute_feature
            self._set_feature_hw(images)

        print("Traing vocabulary")
        if self.do_train_clusters:
            self.train_clusters(low_features)

        print("Extracting BOF feature")
        bof = self.toBOF(low_features)

        print("Training classifier")
        if self.do_train_classifier:
            self.train_classifier(bof, labels)

        self.save_model()

    def test(self, images, labels, use_batch = True, batch_size = 100, precompute_feature = None):

        predict = self.inference(images, use_batch=use_batch, 
                    batch_size=batch_size, precompute_feature= precompute_feature)
        report = classification_report(labels, predict, output_dict = True)
        print(report)
        return report


    def inference(self, images, use_batch = True, batch_size = 100, precompute_feature = None):
        """ inference procedure, able to use batch to accelerate the progress of inference.
        """

        if precompute_feature is None:
            print("Extracting Dense sift feature")
            low_features = self.feature_dense_sift(images)
            if self.save_sift_feature:
                self.save_feature(low_features)
        else:
            print("Using pre-computed feature")
            low_features = precompute_feature
            self._set_feature_hw(images)

        print("Extracting BOF feature")
        bof = self.toBOF(low_features)

        num_samples = bof.shape[0]
        predict = np.zeros((num_samples, ), dtype = int)
        iternum = num_samples // batch_size  + 1

        print("Inference on classifier")
        for i in tqdm(range(iternum)):
            start_idx = batch_size * i 
            end_idx = min(batch_size * (i + 1), num_samples)
            if end_idx - start_idx == 0:
                break
            data = bof[start_idx: end_idx, ]
            out = self.predict_clss(data)
            predict[start_idx: end_idx, ] = out

        return predict


def main():

    from data import Caltech101

    demo_config = {
        "dataset_root" : "~", 
        "using_clss": 8,
        "num_clss_train_image": 15,
        "num_clss_test_image": 20,
        "M": 200,
        "L": 2,
    }

    config = demo_config

    dataset = Caltech101(config["dataset_root"])
    X1, y1, X2, y2 = dataset.xy(using_clss=config["using_clss"], 
                                num_clss_train_image = config["num_clss_train_image"], 
                                num_clss_test_image = config["num_clss_test_image"])

    startime = time.clock()
    try:
        s = SpatialPyramidClassifier(
                M = config["M"], L = config["L"]
            )

        s.train(X1, y1)
        report = s.test(X2, y2)

        # # example of training with precomputed features.
        # train_feature = s.load_feature(s.saveroot + "feature_train.npy")
        # test_feature = s.load_feature(s.saveroot + "feature_test.npy")
        # s.train(X1, y1, precompute_feature=train_feature)
        # report = s.test(X2, y2, precompute_feature=test_feature)

        json_dump(report, s.saveroot + "save" + currentTime() + ".json")
        json_dump(config, s.saveroot + "config" + currentTime() + ".json")

    except Exception as e:
        endtime = time.clock()
        print("Using {} s".format(endtime - startime))
        raise e

    endtime = time.clock()
    print("Using {} s".format(endtime - startime))

if __name__ == "__main__":
    main()