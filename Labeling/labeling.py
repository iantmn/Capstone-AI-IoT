import numpy as np

from collections.abc import Collection, Iterable


class labeling():
    def __init__(self, action_ID: str) -> None:
        self.action_ID = action_ID
        self.labels = self.get_labels()

    def get_labels(self) -> list[str]:
        labels: list[str] = []
        done = False
        while not done:
            label = input(f"Enter a new label for '{self.action_ID}' (enter 'l' to see the current labels and enter 'x' to stop):\n")
            if label == 'l':
                if hasattr(self, 'labels'):
                    print(f"{set(self.labels) & set(labels)}")
                else:
                    print(f"{set(labels)}")
            elif label == 'x':
                done = True
            else:
                labels.append(label)

        return labels
