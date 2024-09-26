import torch
import torchvision.utils as vutils


def save_images(generator, epoch, step, device):
    """
    Generates and saves a batch of images using the GAN generator.

    This function creates a batch of fake images by passing random noise (latent vectors)
    through the generator. The generated images are saved as a single grid in a PNG file.
    The file is named according to the current epoch and training step to help monitor
    the progress of the GAN training.

    Parameters:
    -----------
    generator : Generator
        The trained generator model that produces images from random noise.

    epoch : int
        The current epoch number during training. Used in the file name for saving the images.

    step : int
        The current training step within the epoch. Also used in the file name to keep track of image generation progress.

    device : torch.device
        The device (CPU or GPU) on which the generator and computations are located.

    Returns:
    --------
    None
        The function saves the generated images to a specified file, but does not return any value.

    Notes:
    ------
    - The function generates 64 images by passing random noise of size (100, 1, 1) through the generator.
    - The generated images are saved in a grid layout using `torchvision.utils.save_image`, and they are normalized
      to the range [0, 1] for visualization purposes.
    - The output file is saved in the `./output/images/` directory with a filename that includes the epoch and step number.
    """

    # Generate images using random noise
    noise = torch.randn(64, 100, 1, 1, device=device)
    fake_images = generator(noise)

    # Save the generated images as a grid
    vutils.save_image(
        fake_images, f"./output/images/epoch_{epoch}_step_{step}.png", normalize=True
    )
