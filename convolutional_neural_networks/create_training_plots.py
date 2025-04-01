import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import cycle
import matplotlib.lines as mlines


def create_training_loss_plot(data_folder, abbr, title, out_path='figure.png'):
    data = {}
    for file_name in os.listdir(data_folder):
        file_data = pd.read_csv(os.path.join(data_folder, file_name))
        abbr_val = [x[len(abbr):] for x in file_name.split('_') if x.startswith(abbr)][0]
        if abbr_val not in data:
            data[abbr_val] = []
        data[abbr_val].append(file_data)
    plt.figure(figsize=(10, 6))

    palette = sns.color_palette("husl", n_colors=len(data.keys()))

    for color, key in zip(palette, data.keys()):
        for df in data[key]:
            plt.plot(df['train_loss'], color=color, linestyle='-', alpha=0.7)
            plt.plot(df['test_loss'], color=color, linestyle='--', alpha=0.7)

    legend_handles = []

    legend_handles.append(mlines.Line2D([], [], 
                                    color='none', 
                                    linestyle='',
                                    label=title))

    for color, key in zip(palette, data.keys()):
        legend_handles.append(mlines.Line2D([], [], 
                                        color=color,
                                        linestyle='-',
                                        linewidth=2,
                                        label=key))

    legend_handles.append(mlines.Line2D([], [], 
                                    color='none', 
                                    linestyle='',
                                    label='Dataset type'))

    legend_handles.append(mlines.Line2D([], [], 
                                    color='black', 
                                    linestyle='-',
                                    linewidth=2,
                                    label='Train'))
    legend_handles.append(mlines.Line2D([], [], 
                                    color='black', 
                                    linestyle='--',
                                    linewidth=2,
                                    label='Test'))

    plt.title("Training and Test Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(handles=legend_handles,
            bbox_to_anchor=(1.05, 1),
            loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path)

def generate_plots():
    create_training_loss_plot(os.path.join('final-output', 'learning_rate'), 'lr', 'Learning Rate', os.path.join('final-output', 'lr.png'))
    create_training_loss_plot(os.path.join('final-output', 'momentum'), 'momentum', 'Momentum', os.path.join('final-output', 'momentum.png'))
    create_training_loss_plot(os.path.join('final-output', 'weight_decay'), 'wd', 'Weight Decay', os.path.join('final-output', 'weight_decay.png'))

if __name__ == '__main__':
    generate_plots()