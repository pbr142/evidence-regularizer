from typing import Optional
import tensorflow as tf
import pickle
from pathlib import Path
from tensorflow import keras


def load_labels(name: str, label: str):
    if name == 'cifar10':
        file = Path(__file__).parents[1] / 'CIFAR-N' / 'CIFAR-10_human.pickle'
        with open(file, 'rb') as f:
            labels = pickle.load(f)
            return labels[label]
    elif name == 'cifar100':
        file = Path(__file__).parents[1] / 'CIFAR-N' / 'CIFAR-100_human.pickle'
        with open(file, 'rb') as f:
            labels = pickle.load(f)
            return labels[label]
    else:
        raise ValueError(f'No special labels for dataset: {name}')

        
def _preprocess(x, y, dataset):
    x = (x / 255.0) * 2 - 1
    if 'mnist' in dataset:
        x = x.reshape(-1,28,28,1)
    y = tf.keras.utils.to_categorical(y)
    return x, y


def _data_generator(x, y, batch_size: int = 1000, seed: int = 1234):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.cache()
    data = data.shuffle(len(data), seed=seed)
    data = data.batch(batch_size)
    data = data.prefetch(tf.data.AUTOTUNE)
    return data


def load_data(name: str, label: Optional[str] = None):
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, name).load_data()
    if label is not None:
        y_train = load_labels(name, label)
    x_train, y_train = _preprocess(x_train, y_train, name)
    x_test, y_test = _preprocess(x_test, y_test, name)

    return x_train, y_train, x_test, y_test


def load_data_generator(name: str, batch_size: int = 1024, label: Optional[str] = None, seed: int = 1234):
    x_train, y_train, x_test, y_test = load_data(name, label)
    dtrain = _data_generator(x_train, y_train, batch_size, seed)
    dtest = _data_generator(x_test, y_test, batch_size, seed)
    input_shape = x_train[0].shape
    n_classes = len(y_train[0])
    return dtrain, dtest, input_shape, n_classes


def get_preprocess_layers(dataset: str) -> list:
    if dataset == 'mnist':
        return [
            keras.layers.ZeroPadding2D(padding=4),
            keras.layers.RandomCrop(28, 28),
            ]
    elif dataset == 'fashion_mnist':
        return [
            keras.layers.ZeroPadding2D(padding=4),
            keras.layers.RandomCrop(28, 28),
            keras.layers.RandomFlip('horizontal')
            ]

    elif 'cifar' in dataset:
        return [
            keras.layers.ZeroPadding2D(padding=4),
            keras.layers.RandomCrop(32, 32),
            keras.layers.RandomFlip('horizontal')
            ]
    else:
        raise ValueError(f'No Preprocessing defined for dataset: {dataset}')

    
