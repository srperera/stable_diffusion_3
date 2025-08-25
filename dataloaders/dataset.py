import os
import glob
import torch
from typing import List
from tqdm.notebook import tqdm
from torchvision import datasets
from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset


######################################################################################
######################################################################################
def get_data(num_items: int = 1000) -> List:
    data = load_dataset(
        "gmongaras/CC12M_and_Imagenet21K_Recap_Highqual_256",
        split="train",
        streaming=True,
    )
    subset = []
    for i, example in tqdm(enumerate(data), desc="Loading Data..."):
        subset.append(example)
        if i == num_items:
            break
    return subset


######################################################################################
######################################################################################
class SimpleDataset(Dataset):
    def __init__(self, data, image_processor):
        self.data = data
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        image = data["image"]
        image = self.image_processor(image)
        caption = data["recaption"]
        return image, caption


######################################################################################
######################################################################################
class CIFARWithText(Dataset):
    def __init__(self, train=True, cifar100=False, image_processor=None):
        self.image_processor = image_processor
        self.cifar100 = cifar100

        if cifar100:
            self.dataset = datasets.CIFAR100(
                root="sd3/data",
                train=train,
                download=True,
            )
            self.class_names = self.dataset.classes
        else:
            self.dataset = datasets.FashionMNIST(
                root="sd3/data",
                train=train,
                download=True,
            )
            self.class_names = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label_idx = self.dataset[idx]
        image = self.image_processor(image)
        class_name = self.class_names[label_idx]
        return {
            "image_data": image,
            "text_data": class_name,
            "original_image:": image,
            "original_text": class_name,
        }


######################################################################################
######################################################################################
class ProcessedDataset(Dataset):
    def __init__(self, dataset_path: str):
        # get all the chunks from the dataset path
        self.data = glob.glob(os.path.join(dataset_path, "*.pt"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        data = torch.load(data)
        return {"x": data["x"], "c": data["c"], "c_pooled": data["c_pooled"]}
