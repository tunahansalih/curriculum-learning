import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
import wandb
from sklearn.model_selection import train_test_split


from tqdm import tqdm

from dataset import cifar100
from dataset.cifar100 import CIFAR100
from models.cifar100_model import vgg_st
from scoring import knowledge_transfer


# superclasses = [
#     None,
#     "small_mammals",
#     "aquatic_mammals",
#     "fish",
#     "flowers",
#     "food_containers",
#     "fruit_and_vegetables",
#     "household_electrical_devices",
#     "household_furniture",
#     "insects",
#     "large_carnivores",
#     "large_man-made_outdoor_things",
#     "large_natural_outdoor_scenes",
#     "large_omnivores_and_herbivores",
#     "medium_mammals",
#     "non-insect_invertebrates",
#     "people",
#     "reptiles",
#     "small_mammals",
#     "trees",
#     "vehicles_1",
#     "vehicles_2",
# ]


config = dict(
    # Model Parameters
    model_name="stVGG",
    model_steps=70000,
    model_optimizer="sgd",
    model_weights=None,
    model_validation_step=1000,
    # Dataset Parameters
    dataset="cifar100",
    dataset_superclass=None,
    dataset_batch_size=100,
    # Learning Rate Parameters
    lr_initial=0.12,
    lr_decay_rate=1.1,
    lr_minimum=1e-3,
    lr_decay_step_length=400,
    # Curriculum Parameters
    curriculum="vanilla",
    curriculum_step_length=100,
    curriculum_increase=1.9,
    curriculum_starting_percent=0.04,
    curriculum_order="inception",
    curriculum_balance=True,
)


def run(
    model_name,
    model_steps,
    model_optimizer,
    model_weights,
    model_validation_step,
    dataset,
    dataset_superclass,
    dataset_batch_size,
    lr_initial,
    lr_decay_rate,
    lr_minimum,
    lr_decay_step_length,
    curriculum,
    curriculum_step_length,
    curriculum_increase,
    curriculum_starting_percent,
    curriculum_order,
    curriculum_balance,
):

    if dataset == "cifar100":
        # Load CIFAR-100
        (train_x, train_y), (test_x, test_y) = cifar100.CIFAR100.load_data(
            data_dir=data_dir, superclass=dataset_superclass, normalize=True
        )
        classes = CIFAR100.label_encoder.classes_

    elif dataset == "cifar10":
        raise NotImplementedError("Cifar 10 has not yet been implemented")
    else:
        raise ValueError("Only cifar100 and cifar10 are supported")

    num_classes = len(classes)

    if model_name == "stVGG":
        model = vgg_st(num_classes=num_classes)
        image_size = (32, 32)
    elif model_name == "EfficientNetB0":
        model = tf.keras.applications.EfficientNetB0(
            include_top=True,
            weights=None,
            classes=num_classes,
            classifier_activation="softmax",
        )
        image_size = (224, 224)
    else:
        raise ValueError("Only stVGG amd EfficientNetB0 is implemented")

    if curriculum in ["curriculum", "anti_curriculum"]:
        if curriculum_order == "inception":
            # Load Curriculum Order
            train_score = knowledge_transfer.knowledge_transfer_scoring(
                data_dir=data_dir,
                superclass=dataset_superclass,
                train_x=train_x,
                train_y=train_y,
            )
        else:
            raise ValueError("Only inception is supported")

        if curriculum == "curriculum":
            indices = np.argsort(train_score)[::-1]
        elif curriculum == "anti_curriculum":
            indices = np.argsort(train_score)

    elif curriculum == "random":
        indices = np.arange(len(train_x))

    elif curriculum == "vanilla":
        indices = np.arange(len(train_x))

    else:
        raise ValueError(
            "Only curriculum, vanilla, anti_curriculum and random implemented"
        )

    train_x = train_x[indices]
    train_y = train_y[indices]

    # Split validation Data
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_x, train_y, test_size=0.1, stratify=train_y, random_state=53
    )

    if curriculum in ["curriculum", "anti_curriculum", "random"]:
        if curriculum_balance:

            label_instances = []

            indices = []

            for label in classes:
                label_indices = np.where(train_y == label)
                label_instances.append(label_indices[0])

            balanced_indices = []
            for i in range(len(train_y) // num_classes):
                for j in range(num_classes):
                    balanced_indices.append(label_instances[j][i])

            train_x = train_x[balanced_indices]
            train_y = train_y[balanced_indices]

        # train_generator = cifar100.CIFAR100.load_curriculum_generator(
        #     x=train_x,
        #     y=train_y,
        #     batch_size=dataset_batch_size,
        #     step_length=curriculum_step_length,
        #     increase=curriculum_increase,
        #     starting_percent=curriculum_starting_percent,
        #     image_size=image_size,
        # )
        train_generator = cifar100.CIFAR100CurriculumSequence(
            x=train_x,
            y=train_y,
            batch_size=dataset_batch_size,
            step_length=curriculum_step_length,
            increase=curriculum_increase,
            starting_percent=curriculum_starting_percent,
            image_size=image_size,
        )
    elif curriculum == "vanilla":
        train_generator = cifar100.CIFAR100CurriculumSequence(
            x=train_x,
            y=train_y,
            batch_size=dataset_batch_size,
            step_length=None,
            increase=None,
            starting_percent=None,
            image_size=image_size,
        )
    else:
        raise ValueError(
            "only curriculum, anti_curriculum, random and vanilla is accepted"
        )

    validation_generator = cifar100.CIFAR100Sequence(
        x=validation_x,
        y=validation_y,
        batch_size=dataset_batch_size,
        shuffle=False,
        image_size=image_size,
    )

    test_generator = cifar100.CIFAR100Sequence(
        x=test_x,
        y=test_y,
        batch_size=dataset_batch_size,
        shuffle=False,
        image_size=image_size,
    )

    if model_optimizer == "sgd":
        optimizer = tf.optimizers.SGD(learning_rate=lr_initial, momentum=0)
    elif model_optimizer == "adam":
        optimizer = tf.optimizers.Adam(
            learning_rate=lr_initial,
        )
    else:
        raise ValueError("Only sgd and adam implemented")

    loss_fn = tf.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    if model_weights is not None:
        model.load_weights(model_weights)

    for step in tqdm(range(model_steps)):
        g = train_generator.pacing(step)

        train_x_batch, train_y_batch = train_generator[step]
        output = model.train_on_batch(
            train_x_batch,
            train_y_batch,
        )
        for metric_name, metric in zip(model.metrics_names, output):
            wandb.log({f"training/{metric_name}": metric}, step=step)
        wandb.log(
            {"g": g, "learning_rate": model.optimizer.learning_rate.numpy()}, step=step
        )

        if (step % lr_decay_step_length) == 0 and step != 0:
            lr_decayed = max(
                model.optimizer.learning_rate.numpy() / lr_decay_rate, lr_minimum
            )
            K.set_value(model.optimizer.learning_rate, lr_decayed)

        if step > 0 and step % model_validation_step == 0:
            predictions = []
            total_loss = 0
            for validation_step in tqdm(
                range(int(math.ceil(len(validation_y) / dataset_batch_size)))
            ):
                validation_x_batch, validation_y_batch = validation_generator[
                    validation_step
                ]
                prediction_batch = model(validation_x_batch, training=False).numpy()
                total_loss += loss_fn(validation_y_batch, prediction_batch).numpy()
                for y, prediction in zip(validation_y_batch, prediction_batch):
                    prediction_dict = {
                        label: prob
                        for label, prob in zip(
                            list(CIFAR100.label_encoder.classes_), prediction
                        )
                    }
                    prediction_dict["ground_truth"] = y
                    predictions.append(prediction_dict)

            df = pd.DataFrame(
                predictions,
                columns=["ground_truth"] + list(CIFAR100.label_encoder.classes_),
            )

            df.to_csv(
                os.path.join(artifact_dir, f"validation_predictions_{step}.csv"),
                index=False,
            )
            accuracy = np.mean(validation_y == df.iloc[:, 1:].idxmax(axis=1).values)
            wandb.log({"validation/accuracy": accuracy}, step=step)
            wandb.log(
                {
                    "validation/loss": total_loss
                    / int(math.ceil(len(validation_y) / dataset_batch_size))
                },
                step=step,
            )
            model.save_weights(os.path.join(artifact_dir, "model"))

    model.save_weights(os.path.join(artifact_dir, "model"))
    predictions = []

    for test_step in tqdm(range(int(math.ceil(len(test_y) / dataset_batch_size)))):
        test_x_batch, test_y_batch = test_generator[test_step]
        prediction_batch = model(test_x_batch, training=False).numpy()

        for y, prediction in zip(test_y_batch, prediction_batch):
            prediction_dict = {
                label: prob
                for label, prob in zip(
                    list(CIFAR100.label_encoder.classes_), prediction
                )
            }
            prediction_dict["ground_truth"] = y
            predictions.append(prediction_dict)

    df = pd.DataFrame(
        predictions, columns=["ground_truth"] + list(CIFAR100.label_encoder.classes_)
    )

    df.to_csv(os.path.join(artifact_dir, "test_predictions.csv"), index=False)
    accuracy = np.mean(test_y == df.iloc[:, 1:].idxmax(axis=1).values)
    wandb.log({"test/accuracy": accuracy})
    return accuracy


_run = wandb.init(project="curriculum-learning", config=config)
run_id = _run.id

data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

artifact_dir = os.path.join(log_dir, f"experiment_{str(run_id)}")
if not os.path.exists(artifact_dir):
    os.makedirs(artifact_dir)


run(**wandb.config)