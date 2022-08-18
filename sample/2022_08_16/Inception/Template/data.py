import tensorflow_datasets as tfds
import tensorflow as tf 

class DataLoader:
    def __init__(self):
        self.dataset, self.info = tfds.load(
            "tf_flowers", as_supersvised=True, with_info=True 
        )
        self.batch_size = self.info.splits["train"].num_examples
        self.n_classes = self.info.features["label"].num_classes
        self.datasets.map(partial(self, *args, **keywords))