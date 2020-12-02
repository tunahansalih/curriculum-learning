import math
import os
import pickle

import numpy as np
import tensorflow as tf
from skimage.transform import resize
from sklearn import preprocessing

from dataset import download


class CIFAR100Sequence(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size, shuffle=False, image_size=(299, 299)):
        self.x, self.y = x, y
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        if self.shuffle and idx % math.ceil(len(self.x) / self.batch_size) == 0:
            rand_indices = np.random.permutation(len(self.x))
            self.x = self.x[rand_indices]
            self.y = self.y[rand_indices]

        batch_indices = np.arange(
            idx * self.batch_size, (idx + 1) * self.batch_size
        ) % len(self.x)
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        if self.image_size != (32, 32):
            batch_x = np.array([resize(image, self.image_size) for image in batch_x])
        return batch_x, batch_y


class CIFAR100CurriculumSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x,
        y,
        batch_size,
        step_length=None,
        increase=None,
        starting_percent=None,
        image_size=(299, 299),
    ):
        self.x, self.y = x, np.array(y)
        self.batch_size = batch_size
        self.image_size = image_size
        self.step_length = step_length
        self.increase = increase
        self.starting_percent = starting_percent
        self.current_length = self.pacing(0)

    def __len__(self):
        return int(math.ceil(self.current_length / self.batch_size))

    def pacing(self, idx):
        if (
            self.step_length is not None
            and self.increase is not None
            and self.starting_percent is not None
        ):
            g = int(
                min(
                    self.starting_percent
                    * (self.increase ** math.floor(idx / self.step_length)),
                    1,
                )
                * len(self.x)
            )
        else:
            g = len(self.x)
        return g

    def __getitem__(self, idx):
        # Calculate current pace

        self.current_length = self.pacing(idx)

        # Get uniform random batch
        batch_indices = np.random.choice(self.current_length, self.batch_size)
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        if self.image_size != (32, 32):
            batch_x = np.array([resize(image, self.image_size) for image in batch_x])
        return batch_x, batch_y


class CIFAR100:
    label_encoder = preprocessing.LabelEncoder()

    @classmethod
    def load_data(cls, data_dir, superclass=None, normalize=True):
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        download_dir = os.path.join(data_dir, "CIFAR100/")
        data_dir = os.path.join(download_dir, "cifar-100-python")
        download.maybe_download_and_extract(url=url, download_dir=download_dir)
        train_file = os.path.join(data_dir, "train")
        test_file = os.path.join(data_dir, "test")

        with open(train_file, "rb") as f:
            train = pickle.loads(f.read(), encoding="latin1")

        with open(test_file, "rb") as f:
            test = pickle.loads(f.read(), encoding="latin1")

        train_x = train["data"].astype(np.float)
        train_y = train["fine_labels"]

        test_x = test["data"].astype(np.float)
        test_y = test["fine_labels"]

        if superclass is not None:
            meta_file = os.path.join(data_dir, "meta")

            with open(meta_file, "rb") as f:
                meta = pickle.loads(f.read(), encoding="latin1")

            coarse_labels = meta["coarse_label_names"]
            superclass_index = coarse_labels.index(superclass)

            train_y_coarse = train["coarse_labels"]
            test_y_coarse = test["coarse_labels"]

            train_x = train_x[np.where(np.equal(train_y_coarse, superclass_index))]
            train_y = [
                train_y[i]
                for i in range(len(train_y))
                if train_y_coarse[i] == superclass_index
            ]
            test_x = test_x[np.where(np.equal(test_y_coarse, superclass_index))]
            test_y = [
                test_y[i]
                for i in range(len(test_y))
                if test_y_coarse[i] == superclass_index
            ]

        CIFAR100.label_encoder.fit(train_y)
        train_y = CIFAR100.label_encoder.transform(train_y)
        test_y = CIFAR100.label_encoder.transform(test_y)

        train_x = train_x.reshape(-1, 3, 32, 32)
        train_x = train_x.transpose(0, 2, 3, 1)

        test_x = test_x.reshape(-1, 3, 32, 32)
        test_x = test_x.transpose(0, 2, 3, 1)

        if normalize:
            mean_r = np.mean(train_x[:, :, :, 0])
            mean_g = np.mean(train_x[:, :, :, 1])
            mean_b = np.mean(train_x[:, :, :, 2])

            std_r = np.std(train_x[:, :, :, 0])
            std_g = np.std(train_x[:, :, :, 1])
            std_b = np.std(train_x[:, :, :, 2])

            train_x[:, :, :, 0] = (train_x[:, :, :, 0] - mean_r) / std_r
            train_x[:, :, :, 1] = (train_x[:, :, :, 1] - mean_g) / std_g
            train_x[:, :, :, 2] = (train_x[:, :, :, 2] - mean_b) / std_b

            test_x[:, :, :, 0] = (test_x[:, :, :, 0] - mean_r) / std_r
            test_x[:, :, :, 1] = (test_x[:, :, :, 1] - mean_g) / std_g
            test_x[:, :, :, 2] = (test_x[:, :, :, 2] - mean_b) / std_b

        return (train_x, train_y), (test_x, test_y)

    @staticmethod
    def load_generator(x, y, batch_size, shuffle=False, image_size=(299, 299)):
        cifar100_sequence = CIFAR100Sequence(
            x, y, batch_size, shuffle=shuffle, image_size=image_size
        )
        enqueuer = tf.keras.utils.OrderedEnqueuer(
            cifar100_sequence, use_multiprocessing=False
        )
        enqueuer.start(workers=1, max_queue_size=100)
        return enqueuer.get()

    @staticmethod
    def load_curriculum_generator(
        x,
        y,
        batch_size,
        step_length,
        increase,
        starting_percent,
        image_size=(299, 299),
    ):
        cifar100_sequence = CIFAR100CurriculumSequence(
            x,
            y,
            batch_size,
            step_length,
            increase,
            starting_percent,
            image_size,
        )
        enqueuer = tf.keras.utils.OrderedEnqueuer(
            cifar100_sequence, use_multiprocessing=False
        )
        enqueuer.start(workers=1, max_queue_size=100)
        return enqueuer.get()
