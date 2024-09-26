from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10


def get_dataloader(config):
    """
    Returns a DataLoader for the CIFAR-10 dataset based on the provided configuration.

    This function sets up a DataLoader to load the CIFAR-10 dataset, applying transformations
    such as resizing, normalization, and conversion to tensors. The image size and other
    parameters such as batch size are defined by the configuration provided as input.

    Parameters:
    -----------
    config : dict
        A configuration dictionary containing the following keys:
        - "data" : dict
            - "image_size" : int
                The target size to which CIFAR-10 images will be resized. If the size is 32 (the default size of CIFAR-10),
                no resizing will be applied.
            - "dataset_path" : str
                Path to where the dataset should be stored or loaded from.
            - "download" : bool
                Whether to download the dataset if it is not available locally.
        - "train" : dict
            - "batch_size" : int
                The number of samples per batch to load during training.

    Returns:
    --------
    train_loader : DataLoader
        A PyTorch DataLoader object that yields batches of CIFAR-10 images and labels for training.

    Notes:
    ------
    - The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    - The transformation applied to the dataset includes:
        - Resize: Conditionally applied based on the image_size parameter.
        - ToTensor: Converts the images from PIL format to PyTorch tensors.
        - Normalize: Normalizes the image pixels to the range [-1, 1] with mean and std deviation of 0.5 for each channel.
    """

    # Define the image transformations for CIFAR-10
    transform = transforms.Compose(
        [
            # Resize the images only if a different size than 32x32 is specified
            (
                transforms.Resize(config["data"]["image_size"])
                if config["data"]["image_size"] != 32
                else transforms.Lambda(lambda x: x)  # Identity transform (no resize)
            ),
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize the image pixels to range [-1, 1]
        ]
    )

    # Load the CIFAR-10 dataset with the specified transformations
    train_dataset = CIFAR10(
        root=config["data"]["dataset_path"],
        train=True,
        download=config["data"]["download"],
        transform=transform,
    )

    # Create a DataLoader to provide the dataset in batches
    train_loader = DataLoader(
        train_dataset, batch_size=config["train"]["batch_size"], shuffle=True
    )

    return train_loader
