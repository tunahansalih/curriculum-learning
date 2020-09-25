import csv
import math

import numpy as np
import pandas as pd

# import sacred
import tensorflow as tf
from joblib import dump, load
from sklearn.svm import SVC
from tqdm import tqdm

from dataset import cifar100

batch_size = 10
result_csv_path = "data/cifar100_inception_features.csv"
svm_path = "data/svm.joblib"


(train_x, train_y), (test_x, test_y) = cifar100.CIFAR100.load_data()
num_batches_per_epoch = int(math.ceil(len(train_x) / batch_size))
generator = cifar100.CIFAR100.load_generator(train_x, train_y, batch_size, image_size=(299, 299))
scoringModel = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet", pooling="avg")

with open(result_csv_path, "w", newline="") as write_obj:
    column_names = ["numpy_index", "label",] + [f"feature_{i}" for i in range(scoringModel.output_shape[1])]
    csv_writer = csv.DictWriter(write_obj, fieldnames=column_names,)
    csv_writer.writeheader()
    for i in tqdm(range(num_batches_per_epoch)):
        (x, y) = next(generator)
        results = scoringModel(x)

        for j, (feature, label) in enumerate(zip(results, y)):
            feature_dict = {f"feature_{k}": f for k, f in enumerate(feature.numpy())}
            feature_dict["numpy_index"] = j + i * batch_size
            feature_dict["label"] = label

            csv_writer.writerow(feature_dict)

    df = pd.read_csv(result_csv_path)
    feature_columns = [column for column in df.columns if "feature" in column]
    X = df[feature_columns].values
    y = df["label"].values
    svc = SVC(probability=True)
    svc.fit(X, y)
    dump(svc, svm_path)

    X = df[feature_columns].values
    scores = []
    for batch in tqdm(np.array_split(X, 100)):
        scores_batch = svc.predict_proba(batch)
        scores_batch = np.max(scores_batch, axis=-1)
        scores.extend(scores_batch)

    df["score"] = scores

    df.to_csv(result_csv_path)

