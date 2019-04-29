from pathlib import Path

import tensorflow as tf

IMG_SIZE = 225
DATASET_FILE_NAME = "query_and_triplets.txt"
DATASET_FILES_DIR = Path("./dataset_original_files")
TRAIN_VALIDATION_DISTRIB = 0.15
TRAIN_TEST_DISTRIB = 0.05


def get_hard_images():
    download_if_needed()
    n_triplets = get_n_triplets()
    validation_stop = n_triplets - int(n_triplets * TRAIN_TEST_DISTRIB)
    train_stop = validation_stop - int(n_triplets * TRAIN_VALIDATION_DISTRIB)
    data_types = ["train", "validation", "test"]
    img_strings_dict = {
        data_type: {"anchor": [], "positive": [], "negative": []}
        for data_type in data_types
    }
    data_types_gen = (data_type for data_type in data_types)
    data_type = next(data_types_gen)

    with open(DATASET_FILE_NAME) as dataset_file:
        i = 1
        while True:
            search_query = dataset_file.readline().strip()
            if not search_query:
                break
            query_dir = DATASET_FILES_DIR / search_query / str(i)
            for name in ["anchor", "positive", "negative"]:
                dataset_file.readline()
                img_strings_dict[data_type][name].append(
                    open(query_dir / name, "rb").read()
                )
            if i == train_stop or i == validation_stop:
                data_type = next(data_types_gen)
            i += 1
    return {
        data_type: tf.data.Dataset.from_tensor_slices(tuple(triplet_dict.values())).map(
            preprocess_triplet
        )
        for data_type, triplet_dict in img_strings_dict.items()
    }


def get_easy_images():
    download_if_needed()
    return []


def get_n_triplets():
    n_triplets = 0
    for _ in open(DATASET_FILE_NAME):
        n_triplets += 1
    n_triplets //= 4
    return n_triplets


# TODO first construct easy dataset taking positive image from positive or negative ref and negative image from exterior query
# TODO data augmentation


def preprocess_triplet(anchor, positive, negative):
    return (
        preprocess_image_file(anchor),
        preprocess_image_file(positive),
        preprocess_image_file(negative),
    )


def preprocess_image_file(img_string):
    img = tf.image.decode_jpeg(img_string)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))  # TODO test without resize
    return tf.keras.applications.inception_resnet_v2.preprocess_input(img)


def download_if_needed():
    if DATASET_FILES_DIR.exists():
        return
    n_triplets = get_n_triplets()

    with open(DATASET_FILE_NAME) as dataset_file:
        i = 1
        while True:
            search_query = dataset_file.readline().strip()
            if not search_query:
                break
            print(f"Downloading triplet {i}/{n_triplets}")
            anchor = dataset_file.readline().strip()
            positive = dataset_file.readline().strip()
            negative = dataset_file.readline().strip()
            query_dir = DATASET_FILES_DIR / search_query / str(i)
            for name in ["anchor", "positive", "negative"]:
                tf.keras.utils.get_file(
                    name, eval(name), cache_dir=".", cache_subdir=query_dir
                )
            i += 1
