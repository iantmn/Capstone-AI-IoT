from __future__ import annotations

import numpy as np
import pandas as pd
from os import cpu_count

from sklearn.neighbors import LocalOutlierFactor

from IPython.display import HTML, display


class NoveltyDetection():
    def __init__(self, data_file: str, processed_data_files: str):
        self.data_file = data_file
        self.datapd = pd.read_csv(data_file)
        self.processed_data_files = processed_data_files

    def detect(self, contamination: int = 0.1) -> list[int]:
        """Function that detects anomalies in the data using LocalOutlierFactor

        Args: 
            contamination (int): The percentage of the dataset that is considered an outlier. Defaults to 0.1.

        Returns:
            list[int]: A list of the ids of the novelties
        """        

        # Choosing the model LocalOutlierFactor
        clf = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=contamination, n_jobs=int(cpu_count()*3/4))
        # fit and predict the model
        prediction = clf.fit_predict(self.datapd.iloc[:, 3:])
        # Counting the outliers, which are represented with the value -1
        count = 0
        for value in prediction:
            if value == -1:
                count += 1
        # Saving a list of the outliers ids  
        ids = np.where(prediction == -1)[0]

        time_video = []
        # print(self.datapd)
        for id in ids:
            with open(self.data_file) as f:
                f.readline()
                for _ in range(id):
                    f.readline()
                splitted = f.readline().strip().split(',')
                i = int(splitted[0])
                time = float(splitted[2])
            with open(self.processed_data_files) as f:
                    for line in f:
                        splitted = line.strip().split(',')
                        if int(splitted[1]) <= i <= int(splitted[2]):
                            video = splitted[3]
                            time_video.append([time, video])
                            break

        return time_video

    def play_novelties(self, time_video: list[list[float, str]], window_size: float) -> None:
        """Function to display the video in the output cell. The video starts automatically at the timestamp,
        plays for window_size seconds and then goes back to the timestamp to loop.

        Args:
            video_file (str): relative file-location to the video file.
            timestamp (float): starting point of the window, seen from the start of the video in seconds.
            window_size (float): length of the window in seconds.
        """

        # TODO difference in start recording data and recording video. This works for the walking case...
        # for time_vid in time_video:
        #     time_vid[0] += 2.5
            
        # Function to display HTML code  
        display(HTML(f'''
                <head>
                    <script type="text/javascript">
                    let id = 0;
                    const time_video = {time_video};
                    let time = time_video[0][0];
                    const ws = {window_size};

                    function init_nov() {{
                        let video = document.getElementById("nov");
                        video.currentTime = time;
                        play_nov();
                    }}

                    function play_nov() {{
                        let video = document.getElementById("nov");
                        if (video.currentTime < time || video.currentTime >= time + ws) {{
                            video.currentTime = time;
                        }}
                        video.play();
                        setInterval(function () {{
                            if (video.currentTime >= time + ws) {{
                                video.currentTime = time;
                            }}
                        }}, 1);
                    }}

                    function pause_nov() {{
                        let video = document.getElementById("nov");
                        video.pause();
                    }}

                    function prev_nov() {{
                    if (id >= 0) {{
                            id = id - 1;
                            time = time_video[id][0];
                            let video = document.getElementById("nov");
                            video.setAttribute('src', time_video[id][1]);
                            document.getElementById("content").innerHTML = 'Novelty ' + (id + 1) + ' out of {len(time_video)} at ' + time + 's in ' + time_video[id][1];
                            // document.getElementById("content2").innerHTML = time_video[id][1];
                            // document.getElementById("content3").innerHTML = time;
                            play_nov();
                        }}
                    }}

                    function next_nov() {{
                        if (id < {len(time_video)} - 1) {{
                            id = id + 1;
                            time = time_video[id][0];
                            let video = document.getElementById("nov");
                            video.setAttribute('src', time_video[id][1]);
                            document.getElementById("content").innerHTML = 'Novelty ' + (id + 1) + ' out of {len(time_video)} at ' + time + 's in ' + time_video[id][1];
                            // document.getElementById("content2").innerHTML = time_video[id][1];
                            // document.getElementById("content3").innerHTML = time;
                            play_nov();
                        }}
                    }}

                    </script>
                    <title></title>
                </head>
                <body>
                    <video id="nov" width="500px" src="{time_video[0][1]}" muted></video>
                    <br>
                    <script type="text/javascript">init_nov()</script>
                    <button onClick="play_nov()">Play</button>
                    <button onClick="pause_nov()">Pause</button>
                    <button onClick="prev_nov()">Previous novelty</button>
                    <button onClick="next_nov()">Next novelty</button>
                    <div id="content">Novelty 1 out of {len(time_video)} at {time_video[0][0]}s in {time_video[0][1]}</div>
                    <!--<div id="content2"></div>
                    <div id="content3"></div>-->
                </body>
        '''))


if __name__ == '__main__':
    Activity = 'Walking3'
    ND = NoveltyDetection(fr'..\Preprocessed-data\{Activity}\features_{Activity}_scaled_AL_predictionss.csv',
                        fr'..\Preprocessed-data\{Activity}\processed_data_files.txt')
    novelties = ND.detect()
    # time_video = []
    # for t_v in novelties:
    #     t_v[0] += 2.5
    #     time_video.append(t_v)
    print(novelties)
    # print(len(time_video))
    # ND.play_novelties(novelties)
