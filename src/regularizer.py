import tensorflow as tf


class EvidenceRegularizerLayer(tf.keras.layers.Layer):
    def __init__(self, threshold: float, cutoff: float, strength: float = 1.0, track_metrics: bool = True):
        super().__init__()
        self.threshold = threshold
        self.cutoff = cutoff
        self.strength = strength
        self.track_metrics = track_metrics

    def get_config(self) -> dict:
        return {
            "threshold": self.threshold,
            "cutoff": self.cutoff,
            "strength": self.strength,
        }

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        positive_activation = tf.reduce_sum(tf.maximum(inputs, self.cutoff), axis=0)
        negative_activation = tf.reduce_sum(tf.maximum(-inputs, self.cutoff), axis=0)
        node_activation = tf.minimum(positive_activation, negative_activation)
        regularization = tf.minimum(node_activation - self.threshold, 0.0)

        batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
        self.add_loss(-self.strength / batch_size * tf.reduce_sum(regularization))

        if self.track_metrics:
            layer_name = self._name.split("_")[-1]
            self.add_metric(
                tf.reduce_min(positive_activation),
                name=f"minimal_positive_activation_{layer_name}",
                aggregation="mean",
            )

            self.add_metric(
                tf.reduce_min(negative_activation),
                name=f"minimal_negative_activation_{layer_name}",
                aggregation="mean",
            )

            self.add_metric(
                tf.math.count_nonzero(tf.math.less(node_activation, self.threshold), dtype=tf.float32),
                name=f"number_of_nodes_below_threshold_{layer_name}",
                aggregation="mean",
            )

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
        positive_activation = tf.reduce_sum(tf.maximum(x, self.cutoff), axis=0)
        negative_activation = tf.reduce_sum(tf.maximum(-x, self.cutoff), axis=0)
        node_activation = tf.minimum(positive_activation, negative_activation)
        regularization = tf.minimum(node_activation - self.threshold, 0.0)

        batch_size = tf.cast(tf.shape(x)[0], dtype=tf.float32)
        return -self.strength / batch_size * tf.reduce_sum(regularization)
