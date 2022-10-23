import argparse
import logging
from pathlib import Path
import sys

from src.data import load_data_generator, get_preprocess_layers
from src import models
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, help='Defines the subfolder where experiment results are stored', required=True)
parser.add_argument('--dataset', type=str, help='mnist, fashion_mnist, cifar10, cifar100', default='mnist')
parser.add_argument('--batch_size', type=int, help='Batch size for training and test data', default=1024)
parser.add_argument('--label', type = str, help="'clean_label', 'aggre_label', 'worse_label', 'random_label1', 'random_label2', 'random_label3' for cifar10 or 'clean_label', 'noisy_label', 'noisy_coarse_label', 'clean_coarse_label' for cifar100", default=None)
parser.add_argument('--model', type = str, help="'dense'", default='dense')
parser.add_argument('--units', type=int, help="layer definition", nargs='+', default=None)
parser.add_argument('--threshold_pct', type = str, help='Threshold for the Evidence Regularizer, in pct of batch_size', default=0.1)
parser.add_argument('--cutoff', type = float, help='The cutoff value between positive and negative side of hyperplane', default=0.)
parser.add_argument('--strength', type = float, help='Strength of Evidence Regularizer, 0 to turn off', default=0.)
parser.add_argument('--optimizer', type = str, help='Optimizer to be used', default='Adam')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--preprocess', action='store_true', default=False)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value
parser.add_argument('--optimizer_kwargs', nargs='*', help='Named arguments for optimizer', action=ParseKwargs, default=dict())

args = parser.parse_args()


# files and folders
PATH_EXPERIMENT = Path(__file__).parent.resolve() / 'experiments'
PATH_OUT = PATH_EXPERIMENT / args.dataset / args.run
PATH_OUT.mkdir(parents=True, exist_ok=True)

model_file = PATH_OUT / "model.pb"
history_file = PATH_OUT / "history.csv"
log_file = PATH_OUT / "run.log"


# logging settings
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, 'w+'),
        logging.StreamHandler(sys.stdout),
    ],
)

# load data
dtrain, dtest, input_shape, n_classes = load_data_generator(name=args.dataset, batch_size=args.batch_size, label=args.label)
logging.info(f"Data loaded")

# preprocessing steps
preprocess = get_preprocess_layers(args.dataset) if args.preprocess else None

# load model
threshold = args.threshold_pct * args.batch_size
model = getattr(models, args.model)(dataset=args.dataset, preprocess=preprocess, units=args.units, input_shape=input_shape, n_classes=n_classes, threshold=threshold, cutoff=args.cutoff, strength=args.strength)
logging.info(f"Model loaded")
logging.info(model.summary())

# compile model
optimizer = getattr(keras.optimizers, args.optimizer)(**args.optimizer_kwargs)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# fit model
history = model.fit(dtrain, validation_data=dtest, epochs=args.epochs)

# save model
model.save(model_file)

df_history = pd.DataFrame(history.history)
df_history.to_csv(history_file, index_label='epoch')

loss = df_history[['loss', 'val_loss']].plot()
plt.savefig('loss.png')

acc = df_history[['accuracy', 'val_accuracy']].plot()
plt.savefig('acc.png')

config_file = PATH_OUT / 'config.ini'
with open(config_file, 'w') as f:
    f.write(str(args.__dict__))
