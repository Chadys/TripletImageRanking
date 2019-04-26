from pathlib import Path

import tensorflow as tf

IMG_SIZE = 225
DATASET_FILE_NAME = "query_and_triplets.txt"
DATASET_FILES_DIR = Path("./dataset_original_files")
TRAIN_TEST_DISTRIB = 0.1


def get_train_easy_images():
    download_if_needed()
    n_triplets = get_n_triplets()
    train_stop = n_triplets - (n_triplets * TRAIN_TEST_DISTRIB)
    img_strings_dict = {"anchor": [], "positive": [], "negative": []}

    with open(DATASET_FILE_NAME) as dataset_file:
        i = 1
        while True:
            search_query = dataset_file.readline().strip()
            if not search_query:
                break
            query_dir = DATASET_FILES_DIR / search_query / str(i)
            for name in ["anchor", "positive", "negative"]:
                dataset_file.readline()
                img_strings_dict[name].append(open(query_dir / name, "rb").read())
            if i == train_stop:
                break
            i += 1
    return tf.data.Dataset.from_tensor_slices(tuple(img_strings_dict.values())).map(preprocess_triplet)


def get_train_hard_images():
    download_if_needed()
    return []


def get_test_easy_images():
    download_if_needed()
    return []


def get_test_hard_images():
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
