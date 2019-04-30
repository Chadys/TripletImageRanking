from pathlib import Path

import tensorflow as tf

IMG_SIZE = 225
DATASET_FILE_NAME = "query_and_triplets.txt"
DATASET_FILES_DIR = Path("./dataset_original_files")
TRAIN_VALIDATION_DISTRIB = 0.15
TRAIN_TEST_DISTRIB = 0.05
BATCH_SIZE = 128
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_hard_images():
    download_if_needed()
    n_triplets = get_n_triplets()
    n_test = int(n_triplets * TRAIN_TEST_DISTRIB)
    n_validation = int(n_triplets * TRAIN_VALIDATION_DISTRIB)
    validation_stop = n_triplets - n_test
    train_stop = validation_stop - n_validation
    n_train = train_stop
    data_types = ["train", "validation", "test"]
    len_datas_dict = {"train": n_train, "validation": n_validation, "test": n_test}
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
        data_type: (
            len_datas_dict[data_type],
            create_dataset(tuple(triplet_dict.values())),
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


def create_dataset(features):
    len_data = len(features[0])
    return (
        tf.data.Dataset.from_tensor_slices(features)
        .map(preprocess_triplet, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size=len_data)
        .repeat()
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    # TODO change drop_remainder depending on strategy


# TODO first construct easy dataset taking positive image from positive or negative ref and negative image from exterior query
# TODO data augmentation


def preprocess_triplet(anchor, positive, negative):
    return (
        (
            preprocess_image_file(anchor),
            preprocess_image_file(positive),
            preprocess_image_file(negative),
        ),
        0,  # unused label
    )


def preprocess_image_file(img_string):
    img = tf.image.decode_jpeg(img_string, channels=3)
    img = tf.dtypes.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))  # TODO try different resize methods
    if tf.keras.backend.image_data_format() == "channels_first":
        img = tf.transpose(img, [2, 0, 1])  # convert to CHW
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
