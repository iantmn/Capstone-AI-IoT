from __future__ import annotations

from IPython.display import clear_output, display, HTML
import time

from collections.abc import Collection

class VideoLabeler:
    def __init__(self, labels: Collection[str]) -> None:
        self.labels = list(labels)
        self.html_id = 0

    def labeling(self, video_file: str, timestamp: float, window_size: float, fig_id: int, probs: dict[str, float] = None, process: str = '', video_offset: float = 0) -> str:
        """Function to label a window in a given video at a given timestamp

        Args:
            video_file (str): relative file-location to the video file.
            timestamp (float): starting point of the window, seen from the start of the video in seconds.
            window_size (float): length of the window in seconds.
            probs: (Collection, optional): Probability that the frame showed is the corresponding label. Defaults to
            None.

        Returns:
            str: the name of the selected label.
        """
        # Clear the output of the cell
        clear_output(wait=True)
        # Making sure that the cell is empty by waiting some time
        # time.sleep(0.1)
        # Show and play the video
        self.display_html(video_file, timestamp, window_size, fig_id, video_offset)
        # Making sure that the cell is empty by waiting some time

        time.sleep(0.3)
        if process:
            print(process)
        
        # Selecting the label:
        while True:
            # Printing the prompt
            max_length = 0
            for label in self.labels:
                if len(label) > max_length:
                    max_length = len(label)
            print(f"Enter the index or the name of one of the following labels. Enter 'n' to add a new label, 'x' to discard this sample, and 'd' to delete the previously labeled  sample:")
            # Print the labels and their probabilities of classification if given
            for i, label in enumerate(self.labels):
                if probs:
                    if label in probs:
                        prob = probs[label]
                    else:
                        prob = 0.0
                    print(f'{i + 1}. {label}     {" " * (max_length - len(label))}{prob:.4}')
                else:
                    print(f'{i + 1}. {label}')
            # Get the input from the user
            new_label = input()

            try:
                # Try casting the input to an integer if possible
                index = int(new_label)
                # Checking if the input is in the right range
                if 1 > index or index > len(self.labels):
                    raise IndexError
                # Returning the chosen label if this is the case
                return self.labels[index - 1]
            # If the integer is not in the range of the indices of the label list
            except IndexError:
                print(f'Index {index} out of bounds! Try again')
            # If the input is not in integer
            except ValueError:
                # Check if the label exists
                if new_label in self.labels or new_label == 'x' or new_label == 'r':
                    return new_label
                # If the option to add a new label is chosen
                elif new_label == 'n':
                    while True:
                        # Prompt
                        print('Enter the name of the new label.')
                        new_label = input()
                        # Check if the label already exists
                        if new_label not in self.labels:
                            while True:
                                # Ask whether the new label is correct
                                print(f"Is '{new_label}' correct? y/n")
                                y_n = input()
                                # If yes, append the new label to the labels attribute and return the new label
                                if y_n.lower() == 'y':
                                    self.labels.append(new_label)
                                    return new_label
                                # If no, ask again to input a new label
                                elif y_n.lower() == 'n':
                                    break
                                # Not allowed input
                                else:
                                    print('Invalid answer! Try again')
                        # Return the existing one if it already exists
                        else:
                            return new_label
                # Not allowed input
                else:
                    print('Label does not exist! Try again')

    def display_html(self, video_file: str, timestamp: float, window_size: float, fig_id: int = -1, video_offset: float = 0) -> None:
        """Function to display the video in the output cell. The video starts automatically at the timestamp,
        plays for window_size seconds and then goes back to the timestamp to loop.

        Args:
            video_file (str): relative file-location to the video file.
            timestamp (float): starting point of the window, seen from the start of the video in seconds.
            window_size (float): length of the window in seconds.
            video_offset (float): time in seconds that the video start before the start of the data. Defaults to 0.
        """

        timestamp += video_offset
        # print(timestamp, window_size)
        # Function to display HTML code  
        display(HTML(f'''
            <head>
                <script type="text/javascript">
                    function init_{self.html_id}(){{
                        let timestamp = {timestamp};
                        let window_size = {window_size};
                        let video = document.getElementById("{self.html_id}");
                        video.currentTime = timestamp;
                        play_{self.html_id}();
                    }}

                    function play_{self.html_id}(){{
                        let timestamp = {timestamp};
                        let window_size = {window_size};
                        let video = document.getElementById("{self.html_id}");
                        if (video.currentTime < timestamp || video.currentTime >= timestamp + window_size){{
                            video.currentTime = timestamp;
                        }}
                        video.play();
                        setInterval(function(){{
                            if(video.currentTime >= timestamp + window_size){{
                                video.currentTime = timestamp;
                            }}
                        }},1);
                        <!--video.playbackRate = 2-->
                    }}

                    function pause_{self.html_id}(){{
                        let timestamp = {timestamp};
                        let window_size = {window_size};
                        let video = document.getElementById("{self.html_id}");
                        video.pause();
                    }}
                </script>
            </head>
            <body>
                <div style="display:flex">
                    <div style="flex:1">
                        <video id="{self.html_id}" height="300px" src="{video_file}" muted></video><br>
                        <script type="text/javascript">init_{self.html_id}()</script>
                        <button onclick="play_{self.html_id}()">Play</button>
                        <button onclick="pause_{self.html_id}()">Pause</button>
                    </div>
                    <div style="flex:1">  
                    <img id="image" src="Plots/plot_to_label_{fig_id}.png" height="300px"></img>
                        <script type="text/javascript">
                        if ({fig_id} === -1) {{
                            document.getElementById("image").src = "";
                        }}
                        </script>
                    </div>
                </div>
            </body>
        '''))

        self.html_id += 1
