@property
def train_easy_images():
    return []

@property
def train_hard_images():
    return []

@property
def test_easy_images():
    return []

@property
def test_hard_images():
    return []

# TODO construct TFRecord
# TODO first construct easy dataset taking positive image from positive or negative ref and negative image from exterior query