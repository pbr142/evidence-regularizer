import tensorflow as tf


def evidence_regularization(x, threshold: float, cutoff: float):
    positive_activation = tf.reduce_sum(tf.maximum(x, cutoff), axis=0)
    negative_activation = tf.reduce_sum(tf.maximum(-x, cutoff), axis=0)
    node_activation = tf.minimum(positive_activation, negative_activation)
    regularization = tf.maximum(threshold - node_activation, 0.0)

    return regularization


class EvidenceRegularizerLayer(tf.keras.layers.Layer):
    def __init__(self, threshold: float, cutoff: float, strength: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.cutoff = cutoff
        self.strength = strength

    def get_config(self) -> dict:
        return {
            "threshold": self.threshold,
            "cutoff": self.cutoff,
            "strength": self.strength,
        }

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        regularization = evidence_regularization(x=inputs, threshold=self.threshold, cutoff=self.cutoff)
        batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
        self.add_loss(self.strength / batch_size * tf.reduce_sum(regularization))

        return inputs


class EvidenceRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, threshold: float, cutoff: float, strength: float = 1.0):
        self.threshold = threshold
        self.cutoff = cutoff
        self.strength = strength

    def get_config(self) -> dict:
        return {
            "threshold": self.threshold,
            "cutoff": self.cutoff,
            "strength": self.strength,
        }

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        regularization = evidence_regularization(x=x, threshold=self.threshold, cutoff=self.cutoff)
        return self.strength * tf.reduce_sum(regularization)
