import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluation import phases_report

def training_loss_plot(train_score, filename: str = None):

    epoch = range(1,len(train_score['loss'])+1)

    # Plot configuration
    plt.figure(figsize=(15/2.54, 8/2.54))
    plt.rcParams['font.size'] = 10

    color1 = (0, 0, 0)
    color2 = (1, 1, 1)
    color3 = (159/255, 182/255, 196/255)
    color4 = (125/255, 102/255, 102/255)
    color5 = (153/255, 0, 0)

    # Actual plot
    plt.plot(epoch, train_score['loss'], label='Training Loss', color=color1, marker='^', markerfacecolor=color1)
    plt.plot(epoch, train_score['val_loss'], label='Validation Loss', color=color1, marker='s', markerfacecolor=color2)

    # Title and labels
    #plt.title('Training Model loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Training Loss')
    plt.yscale("log")
    plt.xlim(0, len(train_score['loss'])+1)
    plt.legend(loc='lower center', bbox_to_anchor=(1.25, 0.78))

    # Axis settings
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    x = epoch[-1] * 1.1

    y1 = 30 
    ax.set_ylim(bottom=y1)

    plt.text(x, y1, s='Batch-Größe: 4', fontsize=10)
    y2 = y1 + 10
    y0 = y1

    plt.text(x, y2, s='Training: '+ '{:d}'.format(len(train_score['loss']))+ ' Epochen', fontsize=10)
    y0 = y1
    y1 = y2
    y2 = y1**2 / y0


    plt.text(x, y2, s='Loss: Mean Squared Error', fontsize=10)
    y0 = y1
    y1 = y2
    y2 = y1**2 / y0

    plt.text(x, y2, s='Optimierer: Adam', fontsize=10)
    y0 = y1
    y1 = y2
    y2 = y1**2 / y0

    plt.text(x, y2, s='Modellparameter:', fontsize=10, fontweight='bold')

    plt.grid(True, linewidth=1.0)
    if filename != None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='pdf')

    return plt

def single_prediction_plot(case: int, index: np.array, groundtruth: np.array, prediction: np.array, infusion: np.array = None, error: bool = None, filename: str = None):

    j = np.where(index == case)[0][0]

    end = np.where(groundtruth[j] == 0)[0][0]

    time = np.arange(0, prediction[j,:end].shape[0]) * 10 / 60

    plt.figure(figsize=(12/2.54, 6/2.54))
    plt.rcParams['font.size'] = 10

    # Colors
    color1 = (0, 0, 0)
    color2 = (1, 1, 1)
    color3 = (159/255, 182/255, 196/255)
    color4 = (125/255, 102/255, 102/255)
    color5 = (153/255, 0, 0)

    plt.plot(time,groundtruth[j,:end], label='Ground Truth', color=color1)
    plt.plot(time,prediction[j,:end], label='Prediction', color=color5)
    
    if infusion is not None:
        bolus = time[(infusion[j,:end,:] > 100)[:,0]]
        plt.vlines(bolus, 0, 100, color=color3, linestyle='dashed', label='Propofol Bolus')

    plt.xlabel('Operationszeit $t_{OP}$')
    plt.ylabel('Bispektralindex $BIS$')

    # Axis settings
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # Limits
    ax.set_ylim(bottom=0, top=100)
    ax.set_xlim(left=0, right=250)

    # Einheiten auf x-Achse
    xunit = 'min'
    ticks = ax.get_xticks()
    ticks = [int(tick) for tick in ticks]
    ticks_with_units = [xunit if i == len(ticks) - 2 else ticks[i] for i in range(len(ticks))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_with_units)

    # Einheiten auf y-Achse
    yunit = '--'
    ticks = ax.get_yticks()
    ticks = [int(tick) for tick in ticks]
    ticks_with_units = [yunit if i == len(ticks) - 2 else ticks[i] for i in range(len(ticks))]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks_with_units)

    plt.grid(True, linewidth=1.0)

    ### Show prediction error
    if error is not None:
        error_calc = phases_report(prediction[j:j+1], groundtruth[j:j+1], infusion[j:j+1])[error]

        plt.text(262, 50, s= error + ':', fontweight='bold')
        plt.text(262, 36, s='Gesamt:')
        plt.text(262, 24, s='Einleitung:')
        plt.text(262, 12, s='Narkose:')
        plt.text(262, 0,  s='Ausleitung:')
        plt.text(262, -22,  s='Case-ID:')

        plt.text(330, 36, s='{:.2f}'.format(error_calc['All']))
        plt.text(330, 24, s='{:.2f}'.format(error_calc['Induction']))
        plt.text(330, 12, s='{:.2f}'.format(error_calc['Maintenance']))
        plt.text(330, 0,  s='{:.2f}'.format(error_calc['Recovery']))
        plt.text(330, -22,s=str(case))
    else: 
        sex = cases.loc[case]['sex'].replace("M", "Männlich").replace("F", "Weiblich")
        
        plt.text(262, 50, s='Fallinformationen:', fontweight='bold')
        plt.text(262, 36, s='Case ID: ' + str(case))
        plt.text(262, 24, s='Alter: ' + str(int(cases.loc[case]['age'])) + ' Jahre')
        plt.text(262, 12, s='Geschlecht: ' + sex)
        plt.text(262, 0,  s='BMI: ' + str(cases.loc[case]['bmi']))

    ### Legend
    dy = 0
    if infusion is None:
        dy = 0.1
    plt.legend(loc='lower center', bbox_to_anchor=(1.25, 0.6+dy))

    if filename != None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='pdf')

    return plt

    if filename != None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='pdf')

    return plt

def full_prediction_plot(index, groundtruth, prediction, infusion: np.array = None):
    
    fig, axs = plt.subplots(groundtruth.shape[0], 1, figsize=(10, 100))
    baseline = np.ones(groundtruth.shape) * 41.0

    # Colors
    color1 = (0, 0, 0)
    color2 = (1, 1, 1)
    color3 = (159/255, 182/255, 196/255)
    color4 = (125/255, 102/255, 102/255)
    color5 = (153/255, 0, 0)

    for j, filename in enumerate(index):
        end = np.where(groundtruth[j] == 0)[0][0]
        time = np.arange(0, prediction[j,:end].shape[0]) * 10 / 60

        axs[j].plot(time,groundtruth[j,:end], label='Ground Truth', color=color1)
        axs[j].plot(time,prediction[j,:end], label='Prediction', color=color5)
        
        if infusion is not None:
            bolus = time[(infusion[j,:end,:] > 100)[:,0]]
            axs[j].vlines(bolus, 0, 100, color=color3, linestyle='dashed', label='Propofol Bolus')

        #axs[j].plot(time,baseline[j,:end,:], label='Baseline', color=color4)

        axs[j].legend(loc='lower left', ncol=3)
        axs[j].grid(True, linewidth=1.0)
        axs[j].set_title('Case ID: ' + str(filename))
        axs[j].axis([0,end*10/60,0,100])

def full_histogramm_plot(groundtruth, prediction,  filename: str = None):
    timefilter = (groundtruth == 0.0).flatten()
    groundtruth = groundtruth.flatten()
    groundtruth[timefilter] = np.nan

    prediction = prediction.flatten()
    prediction[timefilter] = np.nan
    #prediction[prediction < 5.0] = np.nan

    # Colors
    color1 = (0, 0, 0)
    color2 = (1, 1, 1)
    color3 = (159/255, 182/255, 196/255)
    color4 = (125/255, 102/255, 102/255)
    color5 = (153/255, 0, 0)

    plt.figure(figsize=(12/2.54, 6/2.54))
    plt.rcParams['font.size'] = 10


    plt.hist(prediction, bins=100, density=True, color=color5, label='Prediction')
    plt.hist(groundtruth, bins=100, density=True, color=color1, label='Ground Truth', alpha=0.5)
    
    plt.xlabel('Bispektralindex $BIS$')
    plt.ylabel('Auftrittshäufigkeit $h$')
    plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.05))


    # Axis settings
    ax = plt.gca()
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0, right=100)

    # Einheiten auf x-Achse
    xunit = '--'
    ticks = ax.get_xticks()
    ticks = [int(tick) for tick in ticks]
    ticks_with_units = [xunit if i == len(ticks) - 2 else ticks[i] for i in range(len(ticks))]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_with_units)

    # Einheiten auf y-Achse
    yunit = '--'
    ticks = ax.get_yticks()
    ticks = [float(tick) for tick in ticks]
    ticks_with_units = [yunit if i == len(ticks) - 2 else ticks[i] for i in range(len(ticks))]
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks_with_units)

    plt.grid(True, linewidth=1.0)

    pred_max = np.nanmax(prediction)
    pred_min = np.nanmin(prediction)
    pred_mean = np.nanmean(prediction)
    pred_std = np.nanstd(prediction)

    ground_max = np.nanmax(groundtruth)
    ground_min = np.nanmin(groundtruth)
    ground_mean = np.nanmean(groundtruth)
    ground_std = np.nanstd(groundtruth)
    
    t = 105

    plt.text(t, 0.065, s='Prediction:', fontweight='bold')
    plt.text(t, 0.053, s='$BIS_{min}  = $' + '${:.1f}$'.format(pred_min)  + '$\quad BIS_{max} = $' + '${:.1f}$'.format(pred_max))
    plt.text(t, 0.041, s='$BIS_{mean} = $' + '${:.1f}$'.format(pred_mean) + '$\quad BIS_{std} = $' + '${:.1f}$'.format(pred_std))

    plt.text(t, 0.024, s='Ground Truth:', fontweight='bold')
    plt.text(t, 0.012, s='$BIS_{min}  = $' + '${:.1f}$'.format(ground_min)  + '$\quad BIS_{max} = $' + '${:.1f}$'.format(ground_max))
    plt.text(t, 0.000, s='$BIS_{mean} = $' + '${:.1f}$'.format(ground_mean) + '$\quad BIS_{std} = $' + '${:.1f}$'.format(ground_std))

    if filename != None:
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, format='pdf')

    return plt