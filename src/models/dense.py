from typing import Optional
from tensorflow import keras
from src.regularizer import EvidenceRegularizerLayer

default_units = dict(mnist=[64, 32], fashion_mnist=[128, 64, 64], cifar10=[512, 256, 128, 64, 32], cifar100=[1024, 512, 256, 128, 64])

def dense(dataset, preprocess: Optional[keras.models.Sequential] = None, units: Optional[list] = None, input_shape=None, n_classes=10, threshold=None, cutoff=None, strength=0.0):

    units = default_units[dataset] if units is None else units

    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
    ])
    if preprocess is not None:
        for layer in preprocess:
            model.add(layer)
    model.add(keras.layers.Flatten())
        
    for u in units:
        model.add(keras.layers.Dense(u, activation='tanh'))
        model.add(EvidenceRegularizerLayer(threshold=threshold, cutoff=cutoff, strength=strength))
    
    model.add(keras.layers.Dense(n_classes))

    return model


