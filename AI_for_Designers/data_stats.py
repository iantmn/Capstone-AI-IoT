import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import plotly.io as pio
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

from collections.abc import Collection


class Stats:
    def __init__(self, data_file: str, labels: Collection) -> None:
        self.data_file = data_file
        self.labels = labels
    
    def get_percetages(self) -> dict[str, float]:
        total = 0
        result = {}
        for label in self.labels:
            result[label] = 0

        with open(self.data_file) as f:
            f.readline()
            for line in f:
                result[line.strip().split(',')[1]] += 1
                total += 1

        for key in result:
            result[key] /= total

        return result

    def print_percentages(self) -> None:
        dct = self.get_percetages()
        print('Percentages per label:')
        for key, item in dct.items():
            print(f' {key}: {item}')

    def get_ghan_chart(self, offset):
        # init_notebook_mode()

        df = pd.read_csv(self.data_file)
        df['duration'] = offset

        fig = go.Figure(
            layout = {
                'barmode': 'stack',
                'xaxis': {'automargin': True},
                'yaxis': {'automargin': True}}#, 'categoryorder': 'category ascending'}}
        )

        colors = {'walking': 'blue', 'running': 'green', 'stairs_up': 'purple', 'stairs_down': 'red'}

        for label, label_df in df.groupby('label'):
            fig.add_bar(x=label_df.duration,
                        y=label_df.label,
                        base=label_df.time,
                        orientation='h',
                        showlegend=True,
                        marker=dict(color=colors[label]),
                        # marker=label,
                        name=label)
            # print(label_df.duration)

        for label, label_df in df.groupby('label'):
            fig.add_bar(x=label_df.duration,
                    y= ['All'] * len(label_df.label),
                    base=label_df.time,
                    orientation='h',
                    showlegend=False,
                    marker=dict(color=colors[label]),
                    # marker=label,
                    name=label)
            
        # iplot(fig)
        pio.write_image(fig, 'Plots/distribution.png', scale=6, width=1080*2)

    def show_gan_chart(self, offset: float) -> None:
        self.get_ghan_chart(offset)
        img = mpimg.imread('Plots/distribution.png')
        imgplot = plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        
# s = Stats(r'..\Preprocessed-data\Walking3\features_Walking3_scaled_AL_predictionss.csv', ['walking', 'running', 'stairs_up', 'stairs_down'])
# print(s.get_percetages())
# s.get_ghan_chart(0.2)
    