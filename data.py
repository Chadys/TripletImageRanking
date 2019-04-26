from pathlib import Path

import tensorflow as tf

IMG_SIZE = 225
DATASET_FILE_NAME = "query_and_triplets.txt"
DATASET_FILES_DIR = Path("./dataset_original_files")
TRAIN_TEST_DISTRIB = 0.1


def get_train_easy_images():
    download_if_needed()
    return []


def get_train_hard_images():
    download_if_needed()
    return []


def get_test_easy_images():
    download_if_needed()
    return []


def get_test_hard_images():
    download_if_needed()
    return []


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
    n_triplets = 0
    for _ in open(DATASET_FILE_NAME):
        n_triplets += 1
    n_triplets //= 4
    # train_stop = n_triplets - (n_triplets * TRAIN_TEST_DISTRIB)

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
                dataset_dict[name].append(
                    open(
                        tf.keras.utils.get_file(
                            name, eval(name), cache_dir=".", cache_subdir=query_dir
                        ),
                        "rb"
                    ).read()
                )
            if (
                len(dataset_dict["anchor"]) == NB_TRIPLET_PER_TFRECORD
                or i == train_stop
            ):
                features_dataset = tf.data.Dataset.from_tensor_slices(
                    tuple(dataset_dict.values())
                )
                features_dataset = features_dataset.map(preprocess_triplet)
                writer = tf.data.experimental.TFRecordWriter(
                        str(tfrecord_dir / f"hard_triplets_{j:02}.tfrecord")
                )
                writer.write(features_dataset)
                j += 1
                dataset_dict = {"anchor": [], "positive": [], "negative": []}
            if i == train_stop:
                tfrecord_dir = TFRECORD_FILES_DIR / "hard" / "test"
                # tfrecord_dir.mkdir()
            i += 1
