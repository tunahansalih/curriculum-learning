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


def knowledge_transfer_scoring(data_dir, superclass, train_x, train_y, batch_size=10):
    if superclass is None:
        result_csv_path = os.path.join(data_dir, "cifar100_inception_features.csv")
        ordered_balanced_result_csv_path = os.path.join(
            data_dir, "cifar100_ordered_balanced_indices.csv"
        )
        svm_path = os.path.join(data_dir, "svm.joblib")
    else:
        superclass_data_dir = os.path.join(data_dir, superclass)
        if not os.path.exists(superclass_data_dir):
            os.makedirs(superclass_data_dir)
        result_csv_path = os.path.join(
            superclass_data_dir, "cifar100_inception_features.csv"
        )
        ordered_balanced_result_csv_path = os.path.join(
            superclass_data_dir, "cifar100_ordered_balanced_indices.csv"
        )
        svm_path = os.path.join(superclass_data_dir, "svm.joblib")

    if not os.path.exists(ordered_balanced_result_csv_path):
        if not os.path.exists(result_csv_path):

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

            df.to_csv(result_csv_path)

        labels = df["label"].values
        label_set, label_counts = np.unique(labels, return_counts=True)

        df_list = []
        for label in label_set:
            df_label = df[df["label"] == label]
            df_label = df_label.sort_values(by="score", ascending=False)
            df_list.append(df_label)

        rows = []
        for i in range(min(label_counts)):
            for j, label in enumerate(label_set):
                rows.append(df_list[j].iloc[i])

        ordered_balanced_df = pd.DataFrame(
            rows, columns=["numpy_index", "label", "score"]
        )
        ordered_balanced_df = ordered_balanced_df.astype(
            dtype={"numpy_index": "int64", "label": "int64", "score": "float64"}
        )
        ordered_balanced_df = ordered_balanced_df.reset_index(drop=True)
        ordered_balanced_df.to_csv(ordered_balanced_result_csv_path)
    else:
        ordered_balanced_df = pd.read_csv(ordered_balanced_result_csv_path)
        ordered_balanced_df = ordered_balanced_df.astype(
            dtype={"numpy_index": "int64", "label": "int64", "score": "float64"}
        )
    return ordered_balanced_df["numpy_index"].values
