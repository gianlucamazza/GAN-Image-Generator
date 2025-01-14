# GAN Image Generator

This project implements a **Generative Adversarial Network (GAN)** to generate synthetic images based on the CIFAR-10 dataset. The generator model learns to create realistic images, while the discriminator learns to distinguish between real and fake images.

## Table of Contents
- [GAN Image Generator](#gan-image-generator)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training the GAN](#training-the-gan)
    - [Training Configuration](#training-configuration)
    - [Training Logs and Images](#training-logs-and-images)
  - [Generating Images](#generating-images)
  - [Project Structure](#project-structure)
  - [Configuration](#configuration)
    - [config/config.yaml](#configconfigyaml)
  - [Dependencies](#dependencies)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

This project trains a GAN on the CIFAR-10 dataset to generate images with a resolution of 32x32 pixels. The architecture consists of:
- **Generator**: A neural network that generates fake images from random noise.
- **Discriminator**: A neural network that tries to distinguish between real images from the CIFAR-10 dataset and fake images generated by the generator.

By training both networks in a min-max game, the generator learns to produce more realistic images, while the discriminator improves in distinguishing between real and fake images.

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/gan-image-generator.git
    cd gan-image-generator
    ```

2. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset**:
    The CIFAR-10 dataset will be automatically downloaded and extracted into the `data/` folder when you run the training script for the first time.

## Dataset

The project uses the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes (e.g., airplanes, cars, birds, cats). It is automatically downloaded by the training script and saved under the `data/` folder.

If you wish to download the dataset manually, you can find it [here](https://www.cs.toronto.edu/~kriz/cifar.html).

## Training the GAN

To train the GAN, you need to run the following command:

```bash
python -m src.train
```

### Training Configuration

The configuration for the training process is stored in `config/config.yaml`. You can modify various hyperparameters such as:
- Learning rate
- Batch size
- Number of epochs
- Latent vector size

Example configuration:
```yaml
train:
  batch_size: 64
  epochs: 50
  learning_rate: 0.0002
  latent_dim: 100

data:
  dataset_name: cifar10
  download: true
  dataset_path: "./data"
  image_size: 32
```

### Training Logs and Images

During training, images generated by the GAN will be saved periodically in the `output/images/` directory. Logs are saved in the `output/logs/` directory.

## Generating Images

Once the model is trained, you can generate new images using the `generate_images.py` script. To generate a set of images:

```bash
python -m src.generate_images
```

By default, this script will:
- Load the pretrained generator model from the `models/` directory.
- Generate a specified number of images using random noise as input.
- Save the images in the `output/generated_images/` directory.

You can customize parameters such as the number of images, latent vector size, and output directory by editing the script or passing arguments.

## Project Structure

The project is organized as follows:

```
.
├── config                  # Configuration files
│   └── config.yaml         # Training and model configuration
├── data                    # Dataset directory
│   └── cifar-10-batches-py # CIFAR-10 dataset (downloaded automatically)
├── models                  # Directory for saving trained models
├── output                  # Output directory for images and logs
│   ├── images              # Generated images during training
│   └── logs                # Training logs
├── src                     # Source code for the project
│   ├── dataloader.py       # Data loading utilities
│   ├── discriminator.py    # Discriminator model
│   ├── generator.py        # Generator model
│   ├── gan.py              # GAN training logic
│   ├── train.py            # Script for training the GAN
│   ├── generate_images.py  # Script to generate images with the trained generator
│   └── utils.py            # Utility functions (e.g., save_images)
├── requirements.txt        # Python dependencies
├── setup.py                # Setup script for the project

```

## Configuration

### config/config.yaml

This file contains various settings related to training and data configuration. Key parameters include:
- **train**: Settings related to training (e.g., batch size, number of epochs, learning rate).
- **data**: Dataset-related settings (e.g., dataset path, image size).

You can modify this file to change training behavior without modifying the source code.

## Dependencies

This project requires the following Python packages, which are listed in the `requirements.txt` file:

- `torch`
- `torchvision`
- `numpy`
- `PyYAML`
- `Pillow`

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of your changes.

Before submitting a pull request, please ensure that your code passes all tests and follows the project's coding style.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
