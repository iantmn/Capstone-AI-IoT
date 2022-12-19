import numpy as np
import cv2
import moviepy.editor

from collections.abc import Collection, Iterable, Set


class Labeling():
    def __init__(self, action_ID: str) -> None:
        self.action_ID = action_ID
        self.labels = set()
        self.get_insert_labels()

    def get_insert_labels(self, labels_to_make: int = -1) -> Set[str]:
        """Function to define new (specific amount of) labels. The labels cannot be the same as labels that are defined before.
        At the end, the new labels are added to the label attribute.

        Args:
            labels_to_make (int, optional): Amount of labels to add. If the integer is smaller than 0, unlimited labels can be added.
            If the integer is larger than 0, that specific amount of labels has to be added. Defaults to -1.

        Returns:
            Set[str]: A set of the new labels is returned
        """

        # Empty set to save the new labels in
        labels: Set[str] = set()

        # Every iterations, the labels_to_make variable will be decreased by one (if the label is allowed), until it is zero. If
        # labels_to_make is smaller than zero, the loop will not be exited automatically.
        while labels_to_make != 0:
            label = input(f"Enter a new label for '{self.action_ID}' (enter 'l' to see the current labels and enter 'x' to stop):\n")
            # If 'l' is entered, a set with all the current labels (old and new) are displayed
            if label == 'l':
                if hasattr(self, 'labels'):
                    print(f"{set(self.labels) | set(labels)}")
                else:
                    print(f"{set(labels)}")
            # If 'x' is entered, the loop is exited, if enough labels are made
            elif label == 'x' and labels_to_make <= 0:
                break
            elif label == 'x':
                print("Not enough labels made yet. Add some more")
            # The label is added to the label set
            else:
                if label in self.labels or label in labels:
                    print("Label already exists. Add a non-existing one")
                else:
                    labels.add(str(label))
                    labels_to_make -= 1

        # The labels attribute is updated
        self.labels |= labels

        return labels

    def change_label(self) -> None:
        """Function to update an existing label"""

        while True:
            # Prompt for choosing the label
            labels = list(self.labels)
            print("Choose the right label from the options below:")
            for i, label in enumerate(labels):
                print(f"\t{i}: {label}")
            # Asking for the index of the label
            index = input("Enter the index of the label\n")

            # Check if the input is an integer and if the integer is an index of the labels list
            try:
                i = int(index)
                # Removing the label that needs to be updated
                self.labels.discard(labels[i])
                # Adding the new label
                new_label = set(input("insert changed label name"))
                self.labels |= self.new_label

                return new_label

            except ValueError:
                print(f"{index} not an integer!")
            except IndexError:
                print(f"Index {index} out of bounds!")
            

    def classify(self) -> str:
        """Function to classify a video with one of the labels that is an attribute of the class. An command prompt will be shown to
        to enter the right label. Also an option to add a new label is shown.

        Returns:
            str: The chosen label will be returned.
        """        
        while True:
            # Prompt for choosing the label
            labels = list(self.labels)
            print("Choose the right label from the options below:")
            for i, label in enumerate(labels):
                print(f"\t{i}: {label}")
            print(f"\t{i+1}: None of the above")

            # Asking for the index of the label
            index = input("Enter the index of the label\n")

            # Check if the label exists, else a new one needs to be made
            try:
                # Cast to integer (or raise a ValueError)
                j = int(index)
                # Check if the option to add a new label is chosen
                if len(labels) == j:
                    new_label = self.get_insert_labels(1)
                    return new_label
                # The chosen label will be returned or a IndexError is raised
                else:
                    return labels[j]
                
            except ValueError:
                print(f"{index} not an integer!")
            except IndexError:
                print(f"Index {index} out of bounds!")


class VideoPlayer():
    def __init__(self) -> None:
        self.width = 540
        self.height = 360

    def play_vid(self, filename: str, timestamp: float = 0, window_size: float = 2) -> None:
        # Start playing the video
        is_closed = False
        while not is_closed:            
            # Create a VideoCapture object for the selected video file
            video = cv2.VideoCapture(filename)

            # Extract the fps and the amount of frames in the video from the meta data
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            # Calculate thte amount of frames in the window
            frames_window =  fps * window_size

            # Check whether the inputed timestamp is larger than the length of the video
            if frame_count / fps <= timestamp:
                raise ValueError("Timestamp is longer than the video!")

            # Set the starting point of the video at the timestamp
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            # Set the amount of frames played to 0. This will be used to check when the window length is played
            frames_played = 0
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    break

                # Display the video frame
                cv2.imshow('Video', frame)
                if cv2.waitKey(int(1/fps * 1000)) & 0xFF == ord('q'):
                    is_closed = True
                    break

                # Check if the window size is played
                if frames_played == frames_window + 1:
                    break
                frames_played += 1

        # Release the video capture object and close the window
        video.release()
        cv2.destroyAllWindows()

    @staticmethod
    def make_clip(filename: str, timestamp: float = 0, window_size: float = 2) -> str:
        clip = moviepy.editor.VideoFileClip(filename).subclip(timestamp, timestamp + window_size)
        output_file = f"{filename.split('.mp4')[0]}_{timestamp}_{window_size}.mp4"
        clip.write_videofile(output_file)
        return output_file
