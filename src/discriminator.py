import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator model for a Generative Adversarial Network (GAN).

    The Discriminator is a convolutional neural network that classifies input images
    as real or fake. It uses several convolutional layers with increasing feature maps,
    combined with LeakyReLU activations and Batch Normalization for stable training.

    Architecture:
    -------------
    - Input: A 3x32x32 RGB image (CIFAR-10 image size).
    - Conv2D layers: Extract features at multiple scales.
    - LeakyReLU: Used to allow small gradients when inputs are negative, preventing dead neurons.
    - BatchNorm2D: Applied after certain convolutional layers to stabilize learning and normalize activations.
    - Sigmoid: Final activation to output a probability (between 0 and 1) indicating whether the image is real or fake.

    Attributes:
    -----------
    main : nn.Sequential
        The main neural network consisting of a series of convolutional layers,
        activation functions, and batch normalization.

    Methods:
    --------
    forward(input):
        Defines the forward pass of the discriminator. Takes an input image, passes it
        through the convolutional layers, and returns a scalar probability indicating
        whether the image is real or fake.
    """

    def __init__(self):
        """
        Initializes the Discriminator model.

        The model is composed of several convolutional layers with increasing numbers of
        output channels. The layers downsample the input images by reducing the spatial
        dimensions while increasing the number of features.
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 3x32x32 (RGB image), Output: 64x16x16
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # Input: 64x16x16, Output: 128x8x8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Input: 128x8x8, Output: 256x4x4
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # Input: 256x4x4, Output: 512x2x2
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer: Reduces to a single scalar (1x1x1), representing real/fake probability
            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            nn.Sigmoid(),  # Output between 0 and 1
        )

    def forward(self, input):
        """
        Forward pass of the Discriminator model.

        Parameters:
        -----------
        input : torch.Tensor
            The input tensor representing a batch of images of shape (batch_size, 3, 32, 32).

        Returns:
        --------
        torch.Tensor
            A tensor of shape (batch_size, 1) representing the probability that each input
            image is real (close to 1) or fake (close to 0).
        """
        output = self.main(input)
        return output.view(-1, 1)  # Flatten the output to (batch_size, 1)
