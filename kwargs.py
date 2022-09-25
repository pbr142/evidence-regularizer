from tensorflow import keras

TYPES = ["no_regularization", "l1", "l2", "l1_l2", "dropout", "evidence_regularizer"]


def cifar10():
    model_kwargs = dict(
        input_shape=(32, 32, 3),
        hidden_units=100,
        hidden_layers=5,
        output_units=10,
    )
    return model_kwargs


def get_config(name: str):
    if name == "cifar10":
        model_kwargs = cifar10()
    else:
        raise ValueError("Unknown name: %s" % name)
    compile_kwargs: dict = dict()
    fit_kwargs = dict(epochs=100)

    return model_kwargs, compile_kwargs, fit_kwargs


def run_model_kwargs(type: str, model_kwargs: dict, evidence_regularizer_kwargs: dict) -> dict:
    if type == "no_regularization":
        pass
    elif type == "l1":
        model_kwargs["kernel_regularizer"] = keras.regularizers.l1()
    elif type == "l2":
        model_kwargs["kernel_regularizer"] = keras.regularizers.l2()
    elif type == "l1_l2":
        model_kwargs["kernel_regularizer"] = keras.regularizers.l1_l2()
    elif type == "dropout":
        model_kwargs["dropout"] = True
    elif type == "evidence_regularizer":
        model_kwargs["evidence_regularizer_kwargs"] = evidence_regularizer_kwargs
    else:
        raise ValueError("Unknown type: %s" % type)

    return model_kwargs
