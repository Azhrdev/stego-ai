# -*- coding: utf-8 -*-
"""
Encoder networks for image steganography.

This module contains neural network architectures for encoding
hidden data into images with minimal visual distortion.
"""

import torch
from torch import nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    """
    Simple encoder for hiding data in images.
    
    This network uses basic convolutional layers to hide data in images.
    It's relatively fast but offers a lower hiding capacity.
    
    Args:
        data_depth (int): Bits per pixel to hide
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        
        # Initial feature extraction
        self.features = nn.Sequential(
            # Input: (N, 3, H, W)
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Prepare network for payload injection
        self.injection = nn.Sequential(
            nn.Conv2d(hidden_size + data_depth, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            # Output layer (residual output)
            nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1),
            nn.Tanh(),  # Limit output range to [-1, 1]
        )
        
    def forward(self, image, payload):
        """
        Forward pass to hide data in image.
        
        Args:
            image (torch.Tensor): Cover image tensor (N, 3, H, W)
            payload (torch.Tensor): Payload data tensor (N, D, H, W)
            
        Returns:
            torch.Tensor: Steganographic image (N, 3, H, W)
        """
        # Extract features from cover image
        features = self.features(image)
        
        # Ensure payload has the same spatial dimensions as features
        if features.shape[2:] != payload.shape[2:]:
            payload = F.interpolate(payload, size=features.shape[2:], mode='nearest')
        
        # Concatenate features and payload
        combined = torch.cat([features, payload], dim=1)
        
        # Generate residual image
        residual = self.injection(combined)
        
        # Add residual to input image (skip connection)
        output = image + residual
        
        # Ensure output is in valid range [-1, 1]
        return torch.clamp(output, -1.0, 1.0)


class ResidualEncoder(nn.Module):
    """
    Encoder using residual blocks for better image quality.
    
    This network uses residual connections to preserve image details
    while hiding data more effectively.
    
    Args:
        data_depth (int): Bits per pixel to hide
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        
        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(3)
        ])
        
        # Payload preparation
        self.payload_prep = nn.Sequential(
            nn.Conv2d(data_depth, hidden_size // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(hidden_size + hidden_size // 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, image, payload):
        """
        Forward pass to hide data in image.
        
        Args:
            image (torch.Tensor): Cover image tensor (N, 3, H, W)
            payload (torch.Tensor): Payload data tensor (N, D, H, W)
            
        Returns:
            torch.Tensor: Steganographic image (N, 3, H, W)
        """
        # Initial features
        features = self.features(image)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            features = block(features)
        
        # Prepare payload
        if features.shape[2:] != payload.shape[2:]:
            payload = F.interpolate(payload, size=features.shape[2:], mode='nearest')
        
        payload_features = self.payload_prep(payload)
        
        # Concatenate image features and payload
        combined = torch.cat([features, payload_features], dim=1)
        
        # Generate output image
        residual = self.final(combined)
        output = image + residual
        
        return torch.clamp(output, -1.0, 1.0)


class DenseEncoder(nn.Module):
    """
    Encoder using dense blocks for higher capacity.
    
    This network uses densely connected layers for better information flow,
    allowing higher capacity message hiding with less visual distortion.
    Inspired by DenseNet architecture.
    
    Args:
        data_depth (int): Bits per pixel to hide
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        
        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Dense encoder blocks
        self.dense_blocks = nn.ModuleList([
            DenseBlock(hidden_size, growth_rate=32, num_layers=4)
            for _ in range(2)
        ])
        
        # Transition layers between dense blocks
        self.transitions = nn.ModuleList([
            TransitionLayer(hidden_size + 32 * 4, hidden_size)
        ])
        
        # Payload preparation
        self.payload_prep = nn.Sequential(
            nn.Conv2d(data_depth, hidden_size // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output layers
        final_features = hidden_size + 32 * 4  # Features after last dense block
        self.final = nn.Sequential(
            nn.Conv2d(final_features + hidden_size // 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, image, payload):
        """
        Forward pass to hide data in image.
        
        Args:
            image (torch.Tensor): Cover image tensor (N, 3, H, W)
            payload (torch.Tensor): Payload data tensor (N, D, H, W)
            
        Returns:
            torch.Tensor: Steganographic image (N, 3, H, W)
        """
        # Extract initial features
        features = self.features(image)
        
        # Apply dense blocks with transitions
        for i, dense_block in enumerate(self.dense_blocks):
            features = dense_block(features)
            if i < len(self.transitions):
                features = self.transitions[i](features)
        
        # Prepare payload
        if features.shape[2:] != payload.shape[2:]:
            payload = F.interpolate(payload, size=features.shape[2:], mode='nearest')
        
        payload_features = self.payload_prep(payload)
        
        # Combine features with payload
        combined = torch.cat([features, payload_features], dim=1)
        
        # Generate final output
        residual = self.final(combined)
        output = image + residual
        
        return torch.clamp(output, -1.0, 1.0)


class UNetEncoder(nn.Module):
    """
    U-Net architecture for high-quality steganography.
    
    This network uses a U-Net-like architecture with skip connections
    between encoder and decoder paths, providing excellent image quality
    preservation while encoding information.
    
    Args:
        data_depth (int): Bits per pixel to hide
        hidden_size (int): Base size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        
        # Encoder path (downsampling)
        self.enc1 = self._make_encoder_block(3, hidden_size)
        self.enc2 = self._make_encoder_block(hidden_size, hidden_size * 2)
        self.enc3 = self._make_encoder_block(hidden_size * 2, hidden_size * 4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_size * 4, hidden_size * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 8),
            nn.Conv2d(hidden_size * 8, hidden_size * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 8),
        )
        
        # Payload integration at bottleneck
        self.payload_prep = nn.Sequential(
            nn.Conv2d(data_depth, hidden_size * 4, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Decoder path (upsampling)
        self.dec3 = self._make_decoder_block(hidden_size * 8 + hidden_size * 4, hidden_size * 4)
        self.dec2 = self._make_decoder_block(hidden_size * 4, hidden_size * 2)
        self.dec1 = self._make_decoder_block(hidden_size * 2, hidden_size)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _make_encoder_block(self, in_channels, out_channels):
        """Create a double convolution block for the encoder path."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create a double convolution block for the decoder path."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, image, payload):
        """
        Forward pass to hide data in image.
        
        Args:
            image (torch.Tensor): Cover image tensor (N, 3, H, W)
            payload (torch.Tensor): Payload data tensor (N, D, H, W)
            
        Returns:
            torch.Tensor: Steganographic image (N, 3, H, W)
        """
        # Encoder path with skip connections
        enc1_out = self.enc1(image)
        enc2_in = self.pool(enc1_out)
        
        enc2_out = self.enc2(enc2_in)
        enc3_in = self.pool(enc2_out)
        
        enc3_out = self.enc3(enc3_in)
        bottleneck_in = self.pool(enc3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        # Prepare payload and integrate at bottleneck
        # Resize payload to match bottleneck dimensions
        p_h, p_w = bottleneck_out.shape[2:]
        payload_resized = F.interpolate(payload, size=(p_h, p_w), mode='nearest')
        payload_features = self.payload_prep(payload_resized)
        
        # Combine bottleneck features with payload
        combined = torch.cat([bottleneck_out, payload_features], dim=1)
        
        # Decoder path with skip connections
        dec3_in = self.upsample(combined)
        dec3_out = self.dec3(torch.cat([dec3_in, enc3_out], dim=1))
        
        dec2_in = self.upsample(dec3_out)
        dec2_out = self.dec2(torch.cat([dec2_in, enc2_out], dim=1))
        
        dec1_in = self.upsample(dec2_out)
        dec1_out = self.dec1(torch.cat([dec1_in, enc1_out], dim=1))
        
        # Generate residual output
        residual = self.final(dec1_out)
        
        # Add residual to input image
        output = image + residual
        
        return torch.clamp(output, -1.0, 1.0)


class AttentionEncoder(nn.Module):
    """
    Encoder using self-attention for advanced steganography.
    
    This network uses self-attention mechanisms to identify optimal areas
    for hiding data with minimal visual impact.
    
    Args:
        data_depth (int): Bits per pixel to hide
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth
        
        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Self-attention module
        self.attention = SelfAttention(hidden_size)
        
        # Payload preparation
        self.payload_prep = nn.Sequential(
            nn.Conv2d(data_depth, hidden_size // 2, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(hidden_size + hidden_size // 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, image, payload):
        """
        Forward pass to hide data in image.
        
        Args:
            image (torch.Tensor): Cover image tensor (N, 3, H, W)
            payload (torch.Tensor): Payload data tensor (N, D, H, W)
            
        Returns:
            torch.Tensor: Steganographic image (N, 3, H, W)
        """
        # Extract features
        features = self.features(image)
        
        # Apply self-attention
        attended_features = self.attention(features)
        
        # Prepare payload
        if attended_features.shape[2:] != payload.shape[2:]:
            payload = F.interpolate(payload, size=attended_features.shape[2:], mode='nearest')
        
        payload_features = self.payload_prep(payload)
        
        # Combine features with payload
        combined = torch.cat([attended_features, payload_features], dim=1)
        
        # Generate final output
        residual = self.final(combined)
        output = image + residual
        
        return torch.clamp(output, -1.0, 1.0)


# Helper modules

class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a skip connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class DenseLayer(nn.Module):
    """Dense layer for DenseNet-like blocks."""
    
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv(self.relu(self.bn1(x)))
        out = torch.cat([x, out], dim=1)
        return out


class DenseBlock(nn.Module):
    """Dense block with multiple densely connected layers."""
    
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """Transition layer between dense blocks."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return out


class SelfAttention(nn.Module):
    """Self-attention module for focusing on important image regions."""
    
    def __init__(self, in_channels):
        super().__init__()
        # Reduced channel dimension for efficiency
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Gamma parameter for controlling attention strength, initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Create query, key, value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Calculate attention map
        energy = torch.bmm(query, key)  # Batch matrix multiply
        attention = F.softmax(energy, dim=2)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # Apply attention with residual connection and learnable weight
        out = self.gamma * out + x
        
        return out