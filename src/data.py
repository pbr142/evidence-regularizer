import tensorflow as tf


def _preprocess(x, y):
    x = (x / 255.0) * 2 - 1
    y = tf.keras.utils.to_categorical(y)
    return x, y


def _data_generator(x, y, batch_size: int = 1000, seed: int = 1234, name=None):
    data = tf.data.Dataset.from_tensor_slices((x, y), name=name)
    data = data.cache()
    data = data.shuffle(len(data), seed=seed)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def load_data(name: str):
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, name).load_data()
    x_train, y_train = _preprocess(x_train, y_train)
    x_test, y_test = _preprocess(x_test, y_test)

    return x_train, y_train, x_test, y_test


def load_data_generator(name: str, batch_size: int = 1000, seed=1234):
    x_train, y_train, x_test, y_test = load_data(name)
    dtrain = _data_generator(x_train, y_train, batch_size, seed, name)
    dtest = _data_generator(x_test, y_test, batch_size, seed)
    return dtrain, dtest

