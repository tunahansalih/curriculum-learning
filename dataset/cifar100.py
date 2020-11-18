import math
import os
import pickle

import numpy as np
import tensorflow as tf
from skimage.transform import resize
from sklearn import preprocessing

from dataset import download


class CIFAR100Sequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=False, image_size=(299, 299)):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def preprocess_image(self, x):
        x = x.reshape(3, 32, 32)
        x = x.transpose(1, 2, 0)
        x = x / 255.0
        # x = x - 0.5
        # x = x * 2.0
        return x

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
        return (
            np.array(
                [
                    resize(self.preprocess_image(image), self.image_size)
                    for image in batch_x
                ]
            ),
            batch_y,
        )


class CIFAR100CurriculumSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        x_set,
        y_set,
        batch_size,
        step_length,
        increase,
        starting_percent,
        image_size=(299, 299),
    ):
        self.x, self.y = x_set, np.array(y_set)
        self.batch_size = batch_size
        self.image_size = image_size
        self.step_length = step_length
        self.increase = increase
        self.starting_percent = starting_percent
        self.current_length = int(starting_percent * len(self.x))

    def __len__(self):
        return int(math.ceil(self.current_length / self.batch_size))

    def preprocess_image(self, x):
        x = x.reshape(3, 32, 32)
        x = x.transpose(1, 2, 0)
        x = x / 255.0
        x = x - 0.5
        x = x * 2.0
        return x

    def __getitem__(self, idx):
        # Calculate current pace
        g = int(
            min(
                self.starting_percent
                * (self.increase ** math.floor(idx / self.step_length)),
                1,
            )
            * len(self.x)
        )
        self.current_length = g

        # Get uniform random batch
        batch_indices = np.random.choice(g, self.batch_size)
        batch_x = self.x[batch_indices]
        batch_y = self.y[batch_indices]
        if self.image_size != (32, 32):
            batch_x = np.array(
                [
                    resize(self.preprocess_image(image), self.image_size)
                    for image in batch_x
                ]
            )
        else:
            batch_x = np.array(
                [
                    self.preprocess_image(image)
                    for image in batch_x
                ]
            )
        return batch_x, batch_y


class CIFAR100:
    label_encoder = preprocessing.LabelEncoder()

    @classmethod
    def load_data(cls, data_dir, superclass=None):
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

        train_x = train["data"]
        train_y = train["fine_labels"]

        test_x = test["data"]
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
