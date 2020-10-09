import os
from dataset import cifar100
from scoring import knowledge_transfer

data_dir = "data"
batch_size = 10

superclasses = [
    None,
    "small_mammals",
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
]


if not os.path.exists(data_dir):
    os.makedirs(data_dir)

for superclass in superclasses:
    (train_x, train_y), (test_x, test_y) = cifar100.CIFAR100.load_data(
        data_dir=data_dir, superclass=superclass
    )
    ordered_indices = knowledge_transfer.knowledge_transfer_scoring(
        data_dir, superclass, train_x, train_y
    )
# (train_x, train_y), (test_x, test_y) = cifar100_curriculum.CIFAR100Curriculum.load_data()
# num_steps = 10000

# generator = cifar100_curriculum.CIFAR100Curriculum.load_generator(
#     train_x, train_y, ordered_indices, batch_size=10, step_length=1000, increase=2, starting_percent=0.1, image_size=(299, 299)
# )

# for i in tqdm(range(num_steps)):
#     (x, y) = next(generator)

#     # TODO training script
