import os
import random


class TrainingData:

    def __init__(self, path: str = None):
        self.dataset = []
        self.labels = None

        if not path:
            # Data will be filled by other means
            return

        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory {path} not found.")

        labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        self.labels = [t for t in labels if t != "other"]

        for label in self.labels:
            topic_dir = os.path.join(path, label)
            text_files = [f for f in os.listdir(topic_dir) if f.endswith(".txt")]
            for file in text_files:
                with open(os.path.join(topic_dir, file), "r") as fd:
                    self.dataset.append({
                        "label": label,
                        "content": fd.read(),
                    })

    def sample(self, n: int) -> "TrainingData":
        new_data = TrainingData()
        new_data.labels = self.labels
        new_data.dataset = random.sample(self.dataset, n)
        return new_data




