import matplotlib.pyplot as plt
import numpy as np

def training_loss_plot(train_score, filename: str = None):
    # Plot configuration
    plt.figure(figsize=(15/2.54, 8/2.54))

    # Actual plot
    plt.plot(train_score['loss'], label='Training Loss', color='k', marker='^', markerfacecolor='k')
    plt.plot(train_score['val_loss'], label='Validation Loss', color='k', marker='s', markerfacecolor='w')

    # Title and labels
    #plt.title('Training Model loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.yscale("log")
    plt.xlim(0, len(train_score['loss']))
    plt.legend()

    # Axis settings
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    plt.grid(True, linewidth=1.0)
    if filename != None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='pdf')

    return plt

def single_prediction_plot(j,caseid, groundtruth, prediction, filename: str = None):

    end = np.where(groundtruth[j] == 0)[0][0]

    time = np.arange(0, prediction[j,:end].shape[0]) * 10 / 60

    baseline = np.ones(groundtruth.shape) * 41.0

    plt.figure(figsize=(14/2.54, 6/2.54))
    plt.rcParams['font.size'] = 10

    # Colors
    color1 = (0, 0, 0)
    color2 = (1, 1, 1)
    color3 = (159/255, 182/255, 196/255)
    color4 = (125/255, 102/255, 102/255)
    color5 = (153/255, 0, 0)

    plt.plot(time,groundtruth[j,:end], label='Ground Truth', color=color1)
    plt.plot(time,prediction[j,:end], label='Prediction', color=color5)
    plt.plot(time,baseline[j,:end,:], label='Baseline', color=color4)

    plt.title('CaseID: '+str(caseid[j]))
    plt.xlabel('Operationszeit $t_{OP}$')
    plt.ylabel('Bispektralindex $BIS$')

    plt.legend(loc='lower center', mode="expand", ncol=3)

    # Axis settings
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Limits
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=np.where(groundtruth[j] == 0)[0][0]*10/60)

    # Einheiten auf x-Achse
    xunit = 'min'
    ticks = ax.get_xticks()
    ticks = [int(tick) for tick in ticks]
    ticks_with_units = [xunit if i == len(ticks) - 2 else ticks[i] for i in range(len(ticks))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_with_units)

    plt.grid(True, linewidth=1.0)

    if filename != None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='pdf')

    return plt

def full_prediction_plot(caseids, groundtruth, prediction, filename: str = None):
    fig, axs = plt.subplots(groundtruth.shape[0], 1, figsize=(10, 100))
    baseline = np.ones(groundtruth.shape) * 41.0

    # Colors
    color1 = (0, 0, 0)
    color2 = (1, 1, 1)
    color3 = (159/255, 182/255, 196/255)
    color4 = (125/255, 102/255, 102/255)
    color5 = (153/255, 0, 0)

    for j, filename in enumerate(caseids):
        end = np.where(groundtruth[j] == 0)[0][0]
        time = np.arange(0, prediction[j,:end].shape[0]) * 10 / 60

    

        axs[j].plot(time,groundtruth[j,:end], label='Ground Truth', color=color1)
        axs[j].plot(time,prediction[j,:end], label='Prediction', color=color5)
        axs[j].plot(time,baseline[j,:end,:], label='Baseline', color=color4)

        axs[j].legend(loc='lower center', mode="expand", ncol=3)
        axs[j].set_title('Case ID: ' + str(filename))
        axs[j].axis([0,end*10/60,0,100])