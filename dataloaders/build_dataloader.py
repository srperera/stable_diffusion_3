from typing import Callable, Dict
from torch.utils.data import Dataset, DataLoader


#############################################################################################
#############################################################################################
def build_dataset(config, train, image_processor) -> Dataset:
    # need to refactor. right now this is a dummy dataset to overfit
    # thats why we pass in the train flag.
    # the dataset class needs to be rewritten for a large scale run
    if config.dataset.dataset_type == "cifar":
        from dataloaders.dataset import CIFARWithText

        return CIFARWithText(train, image_processor=image_processor)
    elif config.dataset.dataset_type == "cifar_preprocessed":
        from dataloaders.dataset import ProcessedDataset

        return ProcessedDataset(
            config.dataset.train_data_path if train else config.dataset.val_data_path,
        )

    else:
        raise ValueError("dataset not supported")


#############################################################################################
#############################################################################################
def build_dataloader(
    config: Dict,
    dataset,
    collate_fn: Callable = None,
    train: bool = True,
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        config
        dataset (_type_): _description_
        collate_fn

    Returns:
        DataLoader: _description_
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle if train else False,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    return dataloader


#############################################################################################
#############################################################################################
