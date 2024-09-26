import torch
import os
import torchvision.utils as vutils
from src.generator import Generator

# Parameters
latent_dim = 100  # Size of the noise vector (must match the one used during training)
num_images = 10  # Number of images to generate
output_dir = "./output/generated_images/"  # Directory to save generated images
model_path = "./models/generator.pth"  # Path to the pretrained generator model

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


def generate_images(generator, device, latent_dim, num_images, output_dir):
    """
    Generates a specified number of images using a pretrained GAN generator.

    The generator model is set to evaluation mode, and for each generated image,
    a random noise vector (latent vector) is sampled from a normal distribution.
    The generator then produces an image from the noise vector, which is saved
    to the specified output directory.

    Parameters:
    -----------
    generator : Generator
        The pretrained generator model that will be used to generate images.

    device : torch.device
        The device on which the generator is located, either 'cpu' or 'cuda' (GPU).

    latent_dim : int
        The dimensionality of the noise vector used as input to the generator.

    num_images : int
        The number of images to generate.

    output_dir : str
        The directory where the generated images will be saved.

    Returns:
    --------
    None
        The function saves images to the specified directory but does not return any value.

    Notes:
    ------
    - The generated images are normalized and saved as PNG files.
    - The generator model should be pretrained and loaded from a checkpoint.
    - No gradients are computed during the image generation process (torch.no_grad is used).
    """

    # Set the generator to evaluation mode (disables dropout and batchnorm behavior)
    generator.eval()

    with torch.no_grad():  # Disable gradient computation for inference
        for i in range(num_images):
            # Sample random noise (latent vector) from a standard normal distribution
            noise = torch.randn(1, latent_dim, 1, 1, device=device)

            # Generate a fake image using the generator
            fake_image = generator(noise)

            # Save the generated image to the output directory
            vutils.save_image(
                fake_image,
                os.path.join(output_dir, f"generated_image_{i + 1}.png"),
                normalize=True,  # Normalize image pixels to the range [0, 1]
            )

    print(f"Generated images saved in {output_dir}")


if __name__ == "__main__":
    """
    The main function loads the pretrained generator model and generates images.

    The generator model is loaded from a checkpoint, and the images are generated based
    on random noise vectors sampled from a normal distribution. The images are saved to
    the specified directory.

    - The device is chosen based on availability (GPU if available, otherwise CPU).
    - The number of images and latent dimension are defined by the parameters at the top of the script.
    """

    # Determine the device to use (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pretrained generator model
    generator = Generator(latent_dim, device).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))

    # Generate and save images
    generate_images(generator, device, latent_dim, num_images, output_dir)
