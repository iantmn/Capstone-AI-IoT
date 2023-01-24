import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import plotly.io as pio
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

from collections.abc import Collection
from copy import deepcopy

class Stats:
    def __init__(self, data_file: str, labels: Collection) -> None:
        self.data_file = data_file
        self.labels = labels
    
    def get_percentage(self) -> dict[str, float]:
        """Get the percentage of each label in the data file

        Returns:
            dict[str, float]: key is the label, value is the percentage
        """        
        total = 0
        result = {}
        for label in self.labels:
            result[label] = 0

        with open(self.data_file) as f:
            f.readline()
            for line in f:
                # if line.strip().split(',')[1] in result: 
                # print(line.strip().split(',')[1])
                result[line.strip().split(',')[1]] += 1
                total += 1
                # else:
                #     result[line.strip().split(',')[1]] = 1
                #     total += 1

        for key in result:
            result[key] /= total

        return result

    def print_percentages(self) -> None:
        """Prints the percentages per label
        """        
        dct = self.get_percentage()
        print('Percentages per label:')
        for key, item in dct.items():
            print(f' {key}: {item}')

    def get_ghan_chart(self, offset: float) -> None:    
        """Get the Gantt chart of the data file

        Args:
            size (float): frame size
            offset (float): offset between the frames
            
        Raises:
            ValueError: Raised when there are more labels than colors in the color_list variable. Code won't work otherwise. Designers can add more colors to the color_list variable or reduce the number of labels.
        """        

        df = pd.read_csv(self.data_file)
        df['duration'] = offset

        fig = go.Figure(
            layout = {
                'barmode': 'stack',
                'xaxis': {'automargin': True},
                'yaxis': {'automargin': True}}#, 'categoryorder': 'category ascending'}}
        )
        fig2 = go.Figure(
            layout = {
                'barmode': 'stack',
                'xaxis': {'automargin': True},
                'yaxis': {'automargin': True}}#, 'categoryorder': 'category ascending'}}
        )
        
        color_list = ['blue', 'green', 'purple', 'red', 'orange', 'yellow', 'pink', 'brown', 'cyan', 'magenta', 'olive', 'teal', 'coral', 'gold', 'lavender', 'lime', 'maroon', 'navy', 'orchid', 'plum', 'salmon', 'tan', 'turquoise']
        
        if len(self.labels) > len(color_list):
            # Raise an error when there are more labels than colors in the color_list variable. Code won't work otherwise. Designers can add more colors to the color_list variable or reduce the number of labels.
            raise ValueError(f'Too many labels, please add more colors to the color_list variable inside the dev code or reduce the number of labels. labels: {len(self.labels)}, colors: {len(color_list)}')
        
        # Creating a dictionary with the labels as keys and the colors as values to use in the marker argument of the add_bar function.
        colors = {}
        for i in range(len(self.labels)):
            colors[self.labels[i]] = color_list[i]

        # print(colors)

        # print(df.columns)
        for label, label_df in df.groupby('label'):
            fig.add_bar(x=label_df.duration,
                        y=label_df.label,
                        base=label_df.ID*offset,
                        orientation='h',
                        showlegend=True,
                        marker=dict(color=colors[label]),
                        # marker=label,
                        name=label)
            # print(label_df.duration)  

        for label, label_df in df.groupby('label'):
            fig2.add_bar(x=label_df.duration,
                    y= ['All'] * len(label_df.label),
                    base=label_df.ID*offset,
                    orientation='h',
                    showlegend=True,
                    marker=dict(color=colors[label]),
                    # marker=label,
                    name=label)
            
        # iplot(fig)
        pio.write_image(fig, 'Plots/distribution.png', scale=6, width=len(df), height=220 + len(self.labels)*50)
        pio.write_image(fig2, 'Plots/distribution_all.png', scale=6, width=len(df), height=220 + (len(self.labels)-2)*20)

    def show_ghan_chart(self, offset: float) -> None:
        """shows the Gantt chart of the data file

        Args:
            offset (float): offset between the bars of the chart
        """        

        self.get_ghan_chart(offset)
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        axes[0].imshow(mpimg.imread('Plots/distribution.png'))
        axes[0].axis('off')
        axes[1].imshow(mpimg.imread('Plots/distribution_all.png'))
        axes[1].axis('off')
        plt.show()
        

# Product = 'Mok'
# labels = ['Still', 'Pick_up', 'Put_down', 'Drinking', 'Walking']
# stats = Stats(fr'../Preprocessed-data/{Product}/features_{Product}_scaled_AL_predictionss.csv', labels)
# # # stats.print_percentages()
# stats.show_ghan_chart(0.1)

    