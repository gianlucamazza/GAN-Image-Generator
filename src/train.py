import torch
import logging
from src.config import load_config
from src.dataloader import get_dataloader
from src.gan import GANTrainer

# Logger configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def check_image_size(dataloader, expected_size):
    """
    Checks that the images loaded by the DataLoader have the expected size.

    This function verifies that the images in the first batch of the DataLoader have
    the expected dimensions (channels, height, width). If the size does not match the expected
    size, a ValueError is raised.

    Parameters:
    -----------
    dataloader : DataLoader
        The DataLoader object containing the image batches.
    
    expected_size : tuple
        A tuple specifying the expected size of the images (channels, height, width).
    
    Raises:
    -------
    ValueError:
        If the images do not match the expected size.
    """
    for batch in dataloader:
        images = batch[0]
        if images.shape[1:] != expected_size:
            raise ValueError(
                f"Expected image size {expected_size}, but got {images.shape[1:]}"
            )
        break  # Only check the first batch
    logging.info(f"Image size check passed. Images are {expected_size}")


if __name__ == "__main__":
    """
    Main entry point for loading the configuration, checking image size, and starting GAN training.

    The script performs the following tasks:
    1. Loads the YAML configuration file.
    2. Sets up the training device (GPU or CPU).
    3. Creates a DataLoader for the training data.
    4. Checks that the images have the correct size.
    5. Initializes the GAN trainer and starts the training process.
    
    If any errors occur (e.g., missing configuration file or incorrect image size),
    they are logged and the script exits with an error code.
    """
    # Load the configuration
    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as e:
        logging.error(e)
        exit(1)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create the DataLoader
    train_loader = get_dataloader(config)

    # Check the image size
    expected_size = (3, 32, 32)  # Update if the expected size is different
    check_image_size(train_loader, expected_size)

    # Initialize and start the trainer
    trainer = GANTrainer(config, device)
    trainer.train(train_loader)
