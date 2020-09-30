import math
import numpy as np
from skimage.transform import resize

import tensorflow as tf

from dataset.cifar100 import CIFAR100

class CIFAR100CurriculumSequence(tf.keras.utils.Sequence):
    def __init__(self, x_set, y_set, ordered_indices, batch_size, step_length, increase, starting_percent, image_size=(299, 299)):
        self.x, self.y = x_set, np.array(y_set)
        self.ordered_indices = ordered_indices
        self.batch_size = batch_size
        self.image_size = image_size
        self.step_length = step_length
        self.increase = increase
        self.starting_percent = starting_percent
        self.current_length = int(starting_percent*len(self.x))
        

    def __len__(self):
        return self.current_length

    def preprocess_image(self, x):
        x = x.reshape(3, 32, 32)
        x = x.transpose(1, 2, 0)
        x = x / 255.0
        x = x - 0.5
        x = x * 2.0
        return x

    def __getitem__(self, idx):
        g = int(min(self.starting_percent*(self.increase **
                                           math.floor(idx/self.step_length)), 1)*len(self.x))
        self.current_length = g
        
        #shuffle fraction
        if idx % self.step_length == 0:
            self.ordered_indices[:g] = np.random.permutation(self.ordered_indices[:g])
        fraction_indices = self.ordered_indices[:g]
        fraction_x = self.x[fraction_indices]
        fraction_y = self.y[fraction_indices]

        batch_indices = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size) % g
        batch_x = fraction_x[batch_indices]
        batch_y = fraction_y[batch_indices]
        return (np.array([resize(self.preprocess_image(image), self.image_size) for image in batch_x]), batch_y)


class CIFAR100Curriculum(CIFAR100):

    @staticmethod
    def load_generator(x, y, ordered_indices, batch_size, step_length, increase, starting_percent, image_size=(299, 299)):
        cifar100_sequence = CIFAR100CurriculumSequence(x, y, ordered_indices, batch_size, step_length, increase, starting_percent, image_size)
        enqueuer = tf.keras.utils.OrderedEnqueuer(cifar100_sequence, use_multiprocessing=False, shuffle=False)
        enqueuer.start(workers=1, max_queue_size=10)
        return enqueuer.get()
