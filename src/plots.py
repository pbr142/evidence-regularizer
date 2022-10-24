import matplotlib.pyplot as plt
import matplotlib as mpl

def replace_labels(histories, name_mapping):
    for key, value in name_mapping.items():
        histories['experiment'][histories['experiment'] == key] = value
    return histories

def filter_epochs(histories, start_epoch=0, end_epoch=100):
    histories = histories[histories['epoch'] >= start_epoch]
    histories = histories[histories['epoch'] < end_epoch]
    return histories

def plot_histories(histories, col_order=None):
    def process_histories(histories, variable, col_order = None):
        df = histories[histories['variable'] == variable].pivot(columns='experiment', index='epoch', values='value')
        if col_order is not None:
            df = df.reindex(columns=col_order)
        df.index.names = ['Epoch']
        return df

    mpl.style.use('seaborn')
    cmap = mpl.cm.get_cmap('tab20')
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    plt.set_cmap(cmap)

    df_train = process_histories(histories, 'loss', col_order = col_order)
    df_test = process_histories(histories, 'val_loss', col_order = col_order)
    df_drop = df_test - df_train

    ax_loss_drop = df_drop.plot(ax=ax[0][0], colormap = 'tab20')
    ax_loss_drop.set_title('Drop in Loss from Training to Test data')
    ax_loss_drop.legend(title=None)
    ax_loss = df_train.plot(ax=ax[0][1], colormap = 'tab20', legend=False)
    ax_loss.set_title('Training Loss')
    ax_val_loss = df_test.plot(ax=ax[0][2], colormap = 'tab20', legend=False)
    ax_val_loss.set_title('Test Loss')

    df_train = process_histories(histories, 'accuracy', col_order = col_order)
    df_test = process_histories(histories, 'val_accuracy', col_order = col_order)
    df_drop = df_test - df_train

    ax_acc_drop = df_drop.plot(ax=ax[1][0], colormap = 'tab20', legend=False)
    ax_acc_drop.set_title('Drop in Accuracy from Training to Test data')
    ax_accuracy = df_train.plot(ax=ax[1][1], colormap = 'tab20', legend=False)
    ax_accuracy.set_title('Training Accuracy')
    ax_val_accuracy = df_test.plot(ax=ax[1][2], colormap = 'tab20', legend=False)
    ax_val_accuracy.set_title('Test Accuracy')

    return fig, ax