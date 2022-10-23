import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_histories(histories):

    mpl.style.use('seaborn')
    cmap = mpl.cm.get_cmap('tab20')
    
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
    plt.set_cmap(cmap)
    df_train = histories[histories['variable'] == 'loss'].pivot(columns='experiment', index='epoch', values='value')
    df_test = histories[histories['variable'] == 'val_loss'].pivot(columns='experiment', index='epoch', values='value')
    df_drop = df_test - df_train

    ax_loss_drop = df_drop.plot(ax=ax[0][0], colormap = 'tab20')
    ax_loss_drop.set_title('Drop in Loss from Training to Test data')
    ax_loss = df_train.plot(ax=ax[0][1], colormap = 'tab20', legend=False)
    ax_loss.set_title('Training Loss')
    ax_val_loss = df_test.plot(ax=ax[0][2], colormap = 'tab20', legend=False)
    ax_val_loss.set_title('Test Loss')

    df_train = histories[histories['variable'] == 'accuracy'].pivot(columns='experiment', index='epoch', values='value')
    df_test = histories[histories['variable'] == 'val_accuracy'].pivot(columns='experiment', index='epoch', values='value')
    df_drop = df_test - df_train

    ax_acc_drop = df_drop.plot(ax=ax[1][0], colormap = 'tab20', legend=False)
    ax_acc_drop.set_title('Drop in Accuracy from Training to Test data')
    ax_accuracy = df_train.plot(ax=ax[1][1], colormap = 'tab20', legend=False)
    ax_accuracy.set_title('Training Accuracy')
    ax_val_accuracy = df_test.plot(ax=ax[1][2], colormap = 'tab20', legend=False)
    ax_val_accuracy.set_title('Test Accuracy')

    return fig, ax