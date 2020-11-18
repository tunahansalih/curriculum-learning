import csv
import math
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from dataset import cifar100
from joblib import dump, load
from sklearn.svm import SVC
from tqdm import tqdm


def knowledge_transfer_scoring(
    data_dir: str,
    superclass: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    batch_size: int = 10,
):

    # Choose Superclass
    if superclass is None:
        result_csv_path = os.path.join(data_dir, "cifar100_inception_features.csv")
        svm_path = os.path.join(data_dir, "svm.joblib")
    else:
        superclass_data_dir = os.path.join(data_dir, superclass)
        if not os.path.exists(superclass_data_dir):
            os.makedirs(superclass_data_dir)
        result_csv_path = os.path.join(
            superclass_data_dir, "cifar100_inception_features.csv"
        )
        svm_path = os.path.join(superclass_data_dir, "svm.joblib")

    # Extract Inception Features
    if os.path.exists(result_csv_path):
        df = pd.read_csv(result_csv_path)
    else:
        generator = cifar100.CIFAR100.load_generator(
            train_x, train_y, batch_size, image_size=(299, 299)
        )
        scoringModel = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", pooling="avg"
        )

        with open(result_csv_path, "w", newline="") as write_obj:
            column_names = [
                "numpy_index",
                "label",
            ] + [f"feature_{i}" for i in range(scoringModel.output_shape[1])]
            csv_writer = csv.DictWriter(
                write_obj,
                fieldnames=column_names,
            )
            csv_writer.writeheader()
            num_batches_per_epoch = int(math.ceil(len(train_x) / batch_size))

            for i in tqdm(range(num_batches_per_epoch)):
                (x, y) = next(generator)
                results = scoringModel(x)

                for j, (feature, label) in enumerate(zip(results, y)):
                    feature_dict = {
                        f"feature_{k}": f for k, f in enumerate(feature.numpy())
                    }
                    feature_dict["numpy_index"] = j + i * batch_size
                    feature_dict["label"] = label

                    csv_writer.writerow(feature_dict)

        df = pd.read_csv(result_csv_path)

    # Calculate Difficulty score with SVM Classifier
    if "score" not in df.columns:
        feature_columns = [column for column in df.columns if "feature" in column]
        X = df[feature_columns].values
        y = df["label"].values
        if not os.path.exists(svm_path):
            svc = SVC(probability=True)
            svc.fit(X, y)
            dump(svc, svm_path)
        else:
            svc = load(svm_path)

        scores = []
        for batch_x, batch_y in tqdm(
            zip(np.array_split(X, 100), np.array_split(y, 100))
        ):
            scores_batch = svc.predict_proba(batch_x)
            scores_batch = scores_batch[np.arange(len(scores_batch)), batch_y]
            scores.extend(scores_batch)

        df["score"] = scores

    df.to_csv(result_csv_path, index=False)

    return df["score"].values
