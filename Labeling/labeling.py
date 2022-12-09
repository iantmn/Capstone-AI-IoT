import numpy as np
import cv2
import time

from collections.abc import Collection, Iterable, Set


class labeling():
    def __init__(self, action_ID: str) -> None:
        self.action_ID = action_ID
        self.labels = set()
        self.get_insert_labels()

    def get_insert_labels(self, labels_to_make: int = -1) -> Set[str]:
        labels: Set[str] = set()
        while labels_to_make != 0:
            label = input(f"Enter a new label for '{self.action_ID}' (enter 'l' to see the current labels and enter 'x' to stop):\n")
            if label == 'l':
                if hasattr(self, 'labels'):
                    print(f"{set(self.labels) | set(labels)}")
                else:
                    print(f"{set(labels)}")
            elif label == 'x' and labels_to_make <= 0:
                break
            elif label == 'x':
                print("Not enough labels made yet. Add some more")
            else:
                if label in self.labels or label in labels:
                    print("Label already exists. Add a non-existing one")
                else:
                    labels.add(str(label))
                    labels_to_make -= 1

        self.labels |= labels
        return labels

    def change_label(self) -> None:
        while True:
            # Prompt for choosing the label
            labels = list(self.labels)
            print("Choose the right label from the options below:")
            for i, label in enumerate(labels):
                print(f"\t{i}: {label}")
            # Asking for the index of the label
            index = input("Enter the index of the label\n")

            try:
                i = int(index)
                self.labels.discard(labels[i])
                new_label = set(input("insert changed label name"))
                self.labels |= self.new_label

                return new_label
            except ValueError:
                print(f"{index} not an integer!")
            except IndexError:
                print(f"Index {index} out of bounds!")
            

    def classify(self) -> str:
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
                j = int(index)
                if len(labels) == j:
                    new_label = self.get_insert_labels(1)
                    return new_label
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

    def play_vid(self, video_name: str, timestamp: float = 0, window_size: float = 2) -> None:
        # Start playing the video
        is_closed = False
        while not is_closed:            
            # Create a VideoCapture object for the selected video file
            video = cv2.VideoCapture(video_name)
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
            if frame_count / fps <= timestamp:
                raise ValueError("Timestamp is longer than the video!")

            video.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            start_time = time.time()
            while video.isOpened():
                ret, frame = video.read()

                if not ret:
                    break

                # Display the video frame
                cv2.imshow('Video', frame)
                if cv2.waitKey(int(1/fps * 1000)) & 0xFF == ord('q'):
                    is_closed = True
                    break

                if time.time() - start_time >= window_size:
                    break

        # Release the video capture object and close the window
        video.release()
        cv2.destroyAllWindows()
