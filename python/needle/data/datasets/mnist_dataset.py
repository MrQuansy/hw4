from typing import List, Optional
from ..data_basic import Dataset
import numpy as np


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename) as file:
            magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
            self.images = np.fromstring(file.read(), dtype=np.uint8).reshape(-1, 784)
        with gzip.open(label_filename) as file:
            magic, n = struct.unpack(">II", file.read(8))
            self.labels = np.fromstring(file.read(), dtype=np.uint8)

        self.images = np.float32(self.images) / 255.0
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if isinstance(index, Iterable | slice):
            images = self.images[index]
            labels = self.labels[index]
            images = images.reshape(-1, 28, 28, 1)

            if self.transforms:
                for transform in self.transforms:
                    for i in range(len(images)):
                        images[i] = transform(images[i])
            return images, labels
        else:
            image = self.images[index]  # shape: (784,)
            label = self.labels[index]
            image = image.reshape(28, 28, 1)

            if self.transforms:
                for transform in self.transforms:
                    image = transform(image)

            return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION
