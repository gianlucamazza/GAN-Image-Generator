import torch.nn as nn


class Generator(nn.Module):
    """
    The Generator model for a Generative Adversarial Network (GAN).

    The generator takes a noise vector (latent space) as input and generates an RGB image
    through a series of transposed convolutional layers. It uses Batch Normalization and ReLU
    activations in the hidden layers, and outputs an image with values normalized between -1 and 1
    using the Tanh activation function in the final layer.

    Architecture:
    -------------
    - The input to the model is a latent vector of size `latent_dim` (typically a randomly sampled noise vector).
    - The model upsamples the input through several ConvTranspose2D layers to reach the desired image size of 32x32 pixels.
    - The final layer outputs a 3-channel image (RGB) with pixel values between -1 and 1.

    Attributes:
    -----------
    device : torch.device
        The device on which the generator will be trained or evaluated (CPU or GPU).

    main : nn.Sequential
        A sequential container of layers that define the generator's architecture.

    Methods:
    --------
    forward(input):
        Defines the forward pass of the generator. Takes a noise vector as input and returns
        the generated image after passing through the transposed convolutional layers.
    """

    def __init__(self, latent_dim, device):
        """
        Initializes the Generator model.

        Parameters:
        -----------
        latent_dim : int
            The size of the latent space (i.e., the dimensionality of the noise vector that is used as input).

        device : torch.device
            The device (CPU or GPU) on which the model will be trained or evaluated.
        """
        super(Generator, self).__init__()
        self.device = device

        # Define the generator architecture as a series of ConvTranspose2D layers
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 -> 512 channels, 4x4 image
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # Layer: 512 channels -> 256 channels, 8x8 image
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # Layer: 256 channels -> 128 channels, 16x16 image
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # Layer: 128 channels -> 64 channels, 32x32 image
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # Output Layer: 64 channels -> 3 channels (RGB), 32x32 image
            nn.ConvTranspose2d(64, 3, 3, 1, 1, bias=False),
            nn.Tanh(),  # Normalize the output image values to the range [-1, 1]
        )

    def forward(self, input):
        """
        Forward pass of the generator.

        Takes the input latent vector and generates an RGB image by passing the vector through
        the transposed convolutional layers.

        Parameters:
        -----------
        input : torch.Tensor
            A tensor of shape (batch_size, latent_dim, 1, 1) representing the latent space vectors.

        Returns:
        --------
        torch.Tensor
            A tensor of shape (batch_size, 3, 32, 32) representing the generated RGB images.
        """
        return self.main(input)
