"""Coordinate-conditioned U-Net autoencoder for 128x128 map renders."""

import torch
from torch import nn
from torch.nn import functional as functional


IMAGE_SIZE = 128
EMBEDDING_DIMENSIONS = 256


class CoordConv2d(nn.Module):
    """Append normalized x/y coordinate planes before applying a convolution."""

    def __init__(self, input_channels, output_channels, **convolution_options):
        super().__init__()
        self.convolution = nn.Conv2d(
            input_channels + 2, output_channels, **convolution_options)

    def forward(self, inputs):
        batch_size, _, height, width = inputs.shape
        x_coordinates = torch.linspace(
            -1, 1, width, device=inputs.device, dtype=inputs.dtype)
        y_coordinates = torch.linspace(
            -1, 1, height, device=inputs.device, dtype=inputs.dtype)
        y_grid, x_grid = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
        coordinates = torch.stack((x_grid, y_grid)).expand(batch_size, -1, -1, -1)
        return self.convolution(torch.cat((inputs, coordinates), dim=1))


class DoubleConvolution(nn.Module):
    """Two convolution, normalization, and activation steps."""

    def __init__(self, input_channels, output_channels, coordinate_input=False):
        super().__init__()
        first_convolution = (
            CoordConv2d(input_channels, output_channels, kernel_size=3, padding=1)
            if coordinate_input else
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )
        self.layers = nn.Sequential(
            first_convolution,
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class CoordinateAutoencoder(nn.Module):
    """A 128x128 U-Net with CoordConv and a 256-dimensional latent bottleneck."""

    def __init__(self, input_channels=3, embedding_dimensions=EMBEDDING_DIMENSIONS):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.encoder_128 = DoubleConvolution(input_channels, 32, coordinate_input=True)
        self.encoder_64 = DoubleConvolution(32, 64)
        self.encoder_32 = DoubleConvolution(64, 128)
        self.encoder_16 = DoubleConvolution(128, 256)
        self.bottleneck = DoubleConvolution(256, 256, coordinate_input=True)
        self.pool = nn.MaxPool2d(2)

        self.to_embedding = nn.Linear(256 * 8 * 8, embedding_dimensions)
        self.from_embedding = nn.Linear(embedding_dimensions, 256 * 8 * 8)

        self.up_16 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.decoder_16 = DoubleConvolution(512, 256)
        self.up_32 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_32 = DoubleConvolution(256, 128)
        self.up_64 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_64 = DoubleConvolution(128, 64)
        self.up_128 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder_128 = DoubleConvolution(64, 32)
        self.skip_dropout = nn.Dropout2d(p=0.9999)
        self.output = nn.Conv2d(32, input_channels, kernel_size=1)

    def encode(self, images):
        """Encode a 128x128 batch to its 256-dimensional embedding."""
        self._validate_images(images)
        encoded_128 = self.encoder_128(images)
        encoded_64 = self.encoder_64(self.pool(encoded_128))
        encoded_32 = self.encoder_32(self.pool(encoded_64))
        encoded_16 = self.encoder_16(self.pool(encoded_32))
        bottleneck = self.bottleneck(self.pool(encoded_16))
        embedding = self.to_embedding(bottleneck.flatten(start_dim=1))
        return embedding, (encoded_128, encoded_64, encoded_32, encoded_16)

    # def decode(self, embedding):
    #     """Decode exclusively from the 256-dimensional embedding."""
    #     bottleneck = self.from_embedding(embedding).unflatten(1, (256, 8, 8))
    #     decoded_16 = self.decoder_16(self.up_16(bottleneck))
    #     decoded_32 = self.decoder_32(self.up_32(decoded_16))
    #     decoded_64 = self.decoder_64(self.up_64(decoded_32))
    #     decoded_128 = self.decoder_128(self.up_128(decoded_64))
    #     return self.output(decoded_128)

    def decode(self, embedding, encoder_features):
        """Decode with independently unreliable encoder skip connections."""
        encoded_128, encoded_64, encoded_32, encoded_16 = encoder_features
        bottleneck = self.from_embedding(embedding).unflatten(1, (256, 8, 8))
        decoded_16 = self.decoder_16(torch.cat((
            self.up_16(bottleneck), self.skip_dropout(encoded_16)), dim=1))
        decoded_32 = self.decoder_32(torch.cat((
            self.up_32(decoded_16), self.skip_dropout(encoded_32)), dim=1))
        decoded_64 = self.decoder_64(torch.cat((
            self.up_64(decoded_32), self.skip_dropout(encoded_64)), dim=1))
        decoded_128 = self.decoder_128(torch.cat((
            self.up_128(decoded_64), self.skip_dropout(encoded_128)), dim=1))
        return self.output(decoded_128)

    def forward(self, images):
        embedding, encoder_features = self.encode(images)
        reconstruction = self.decode(embedding, encoder_features)
        return reconstruction, embedding

    @staticmethod
    def _validate_images(images):
        if images.ndim != 4 or images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError("Expected images shaped (batch, channels, 128, 128)")


def soft_sobel_edges(images, threshold=0.2, temperature=15.0):
    """Differentiable Sobel magnitude with a sigmoid edge threshold."""
    grayscale = images.mean(dim=1, keepdim=True)
    sobel_x = images.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))).reshape(1, 1, 3, 3)
    sobel_y = images.new_tensor(((-1, -2, -1), (0, 0, 0), (1, 2, 1))).reshape(1, 1, 3, 3)
    gradient_x = functional.conv2d(grayscale, sobel_x, padding=1)
    gradient_y = functional.conv2d(grayscale, sobel_y, padding=1)
    magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-8)
    magnitude = magnitude / magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    return torch.sigmoid(temperature * (magnitude - threshold))


def trajectory_color_mask(targets, color_threshold=3):
    """Select RGB trajectory pixels using the OpenCV diagnostic filter rule."""
    threshold = color_threshold / 255.0
    red, green, blue = targets.unbind(dim=1)
    return ((red + blue >= 1.0 - threshold) & (green <= threshold)).unsqueeze(1)


def trajectory_color_mse(reconstruction, targets, color_threshold=3):
    """Return RGB MSE over the target pixels used for the trajectory gradient."""
    mask = trajectory_color_mask(targets, color_threshold).expand_as(targets)
    selected_squared_error = (reconstruction - targets).square() * mask
    return selected_squared_error.sum() / mask.sum().clamp_min(1)


def trajectory_color_mae(reconstruction, targets, color_threshold=3):
    """Return RGB MAE over the target pixels used for the trajectory gradient."""
    mask = trajectory_color_mask(targets, color_threshold).expand_as(targets)
    selected_absolute_error = (reconstruction - targets).abs() * mask
    return selected_absolute_error.sum() / mask.sum().clamp_min(1)


def reconstruction_loss(
    reconstruction, targets, pixel_weight=1.0, edge_weight=1.0,
    trajectory_color_weight=1.0, edge_temperature=15.0):
    """Combine image, soft-edge, and heavily weighted trajectory-color MSE."""
    if reconstruction.shape != targets.shape:
        raise ValueError("Reconstruction and target shapes must match")
    pixel_loss = functional.mse_loss(reconstruction, targets)
    edge_loss = functional.mse_loss(
        soft_sobel_edges(reconstruction, temperature=edge_temperature),
        soft_sobel_edges(targets, temperature=edge_temperature))
    trajectory_loss = trajectory_color_mse(reconstruction, targets)
    return (
        pixel_weight * pixel_loss
        + edge_weight * edge_loss
        + trajectory_color_weight * trajectory_loss
    )


# def reconstruction_loss(
#     reconstruction, targets, pixel_weight=1.0, edge_weight=1.0,
#     trajectory_color_weight=1.0, edge_temperature=15.0):
#     """Combine image, soft-edge, and trajectory-color mean absolute errors."""
#     if reconstruction.shape != targets.shape:
#         raise ValueError("Reconstruction and target shapes must match")
#     pixel_loss = functional.l1_loss(reconstruction, targets)
#     edge_loss = functional.l1_loss(
#         soft_sobel_edges(reconstruction, temperature=edge_temperature),
#         soft_sobel_edges(targets, temperature=edge_temperature))
#     trajectory_loss = trajectory_color_mae(reconstruction, targets)
#     return (
#         pixel_weight * pixel_loss
#         + edge_weight * edge_loss
#         + trajectory_color_weight * trajectory_loss
#     )