import torch
import torch.optim as optim
from src.generator import Generator
from src.discriminator import Discriminator
from src.utils import save_images


class GANTrainer:
    """
    A class to manage the training process of a Generative Adversarial Network (GAN).

    This class encapsulates the training loop of a GAN, handling the forward and backward
    passes for both the generator and the discriminator. It uses a binary cross-entropy loss
    function to guide the learning of both models, and optimizes using Adam. Images generated
    during training can be saved at regular intervals.

    Attributes:
    -----------
    config : dict
        A dictionary containing hyperparameters and settings for the training process.
        This typically includes batch size, learning rate, number of epochs, and latent dimension.

    device : torch.device
        The device on which the models and data are located, either "cpu" or "cuda" (GPU).

    latent_dim : int
        The size of the latent space (i.e., the dimensionality of the noise vector used by the generator).

    generator : Generator
        The generator model, which learns to create realistic images from random noise.

    discriminator : Discriminator
        The discriminator model, which learns to classify images as real or fake.

    criterion : torch.nn.BCELoss
        The loss function used to guide both the generator and discriminator. It computes the binary cross-entropy between
        the predicted probabilities and the true labels.

    optimizerG : torch.optim.Adam
        The optimizer for the generator, using the Adam optimization algorithm.

    optimizerD : torch.optim.Adam
        The optimizer for the discriminator, using the Adam optimization algorithm.

    Methods:
    --------
    train(train_loader):
        Executes the training loop for the GAN. For each epoch, the discriminator is updated based on real and fake images,
        and then the generator is updated based on its ability to fool the discriminator. The training process is logged
        and images are saved periodically.
    """

    def __init__(self, config, device):
        """
        Initializes the GANTrainer with the provided configuration and device.

        Parameters:
        -----------
        config : dict
            A configuration dictionary containing hyperparameters such as latent dimension, learning rate,
            batch size, and number of epochs.

        device : torch.device
            The device to use for training (CPU or GPU).
        """
        self.config = config
        self.device = device
        self.latent_dim = config["train"]["latent_dim"]

        # Initialize the generator and discriminator models
        self.generator = Generator(self.latent_dim, self.device).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # Binary Cross-Entropy loss function for both models
        self.criterion = torch.nn.BCELoss()

        # Optimizers for the generator and discriminator
        self.optimizerG = optim.Adam(
            self.generator.parameters(), lr=config["train"]["learning_rate"]
        )
        self.optimizerD = optim.Adam(
            self.discriminator.parameters(), lr=config["train"]["learning_rate"]
        )

    def train(self, train_loader):
        """
        Executes the training loop over the dataset.

        For each batch of images in the training data, this method:
        - Updates the discriminator by computing the loss for both real and fake images.
        - Updates the generator by trying to fool the discriminator with fake images.
        - Logs the losses and saves generated images at regular intervals.

        Parameters:
        -----------
        train_loader : DataLoader
            A PyTorch DataLoader that provides batches of images from the CIFAR-10 dataset.
        """
        for epoch in range(self.config["train"]["epochs"]):
            for i, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                batch_size = images.size(0)

                # Real and fake labels (with label smoothing for real images)
                real_labels = torch.full(
                    (batch_size, 1), 0.9, device=self.device
                )  # Label smoothing for real images
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # --- Update Discriminator ---
                self.optimizerD.zero_grad()  # Reset the gradients for the discriminator

                # Compute loss for real images
                real_outputs = self.discriminator(images)
                loss_real = self.criterion(real_outputs, real_labels)
                loss_real.backward()

                # Generate fake images and compute loss for them
                noise = torch.randn(
                    batch_size, self.latent_dim, 1, 1, device=self.device
                )
                fake_images = self.generator(noise)
                fake_outputs = self.discriminator(
                    fake_images.detach()
                )  # Detach to avoid backpropagation to generator
                loss_fake = self.criterion(fake_outputs, fake_labels)
                loss_fake.backward()

                # Optimize the discriminator
                self.optimizerD.step()

                # --- Update Generator ---
                self.optimizerG.zero_grad()  # Reset the gradients for the generator

                # Compute loss for the generator (we want the discriminator to think these fake images are real)
                output = self.discriminator(fake_images)
                loss_gen = self.criterion(
                    output, real_labels
                )  # The target is real labels
                loss_gen.backward()
                self.optimizerG.step()

                # Log and save images every 100 steps
                if i % 100 == 0:
                    print(
                        f'Epoch [{epoch}/{self.config["train"]["epochs"]}], Step [{i}], '
                        f"Loss_D: {loss_real.item() + loss_fake.item()}, Loss_G: {loss_gen.item()}"
                    )
                    save_images(self.generator, epoch, i, self.device)
