# -*- coding: utf-8 -*-
"""
Decoder networks for image steganography.

This module contains neural network architectures for extracting
hidden data from steganographic images.
"""

import torch
from torch import nn
import torch.nn.functional as F


class BaseDecoder(nn.Module):
    """
    Base class for all decoder networks.
    
    Args:
        data_depth (int): Bits per pixel to extract
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__()
        self.data_depth = data_depth


class SimpleDecoder(BaseDecoder):
    """
    Simple decoder for extracting hidden data from images.
    
    This network uses conventional convolutional layers to extract
    hidden message bits from steganographic images.
    
    Args:
        data_depth (int): Bits per pixel to extract
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__(data_depth, hidden_size)
        
        # Initial feature extraction
        self.features = nn.Sequential(
            # Input: (N, 3, H, W)
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
        )
        
        # Message extraction
        self.extraction = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            # Output layer (raw logits, will be processed with sigmoid during training)
            nn.Conv2d(hidden_size, data_depth, kernel_size=3, padding=1),
        )
        
    def forward(self, stego_image):
        """
        Forward pass to extract hidden data from image.
        
        Args:
            stego_image (torch.Tensor): Steganographic image tensor (N, 3, H, W)
            
        Returns:
            torch.Tensor: Extracted message bits as logits (N, D, H, W)
        """
        # Feature extraction
        features = self.features(stego_image)
        
        # Extract message
        message = self.extraction(features)
        
        return message


class DenseDecoder(BaseDecoder):
    """
    Dense decoder for high-capacity steganography extraction.
    
    This decoder uses densely connected layers to extract hidden information
    more effectively, especially for higher data depths.
    
    Args:
        data_depth (int): Bits per pixel to extract
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__(data_depth, hidden_size)
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Dense blocks
        self.dense1 = DenseBlock(hidden_size, growth_rate=32, num_layers=4)
        self.trans1 = TransitionLayer(hidden_size + 32 * 4, hidden_size)
        self.dense2 = DenseBlock(hidden_size, growth_rate=32, num_layers=4)
        
        # Final extraction
        final_features = hidden_size + 32 * 4  # Features after last dense block
        self.extraction = nn.Sequential(
            nn.Conv2d(final_features, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 2),
            
            # Output layer
            nn.Conv2d(hidden_size // 2, data_depth, kernel_size=3, padding=1),
        )
        
    def forward(self, stego_image):
        """
        Forward pass to extract hidden data from image.
        
        Args:
            stego_image (torch.Tensor): Steganographic image tensor (N, 3, H, W)
            
        Returns:
            torch.Tensor: Extracted message bits as logits (N, D, H, W)
        """
        # Initial features
        features = self.initial(stego_image)
        
        # Dense processing
        features = self.dense1(features)
        features = self.trans1(features)
        features = self.dense2(features)
        
        # Extract message
        message = self.extraction(features)
        
        return message


class UNetDecoder(BaseDecoder):
    """
    U-Net based decoder for high-quality extraction.
    
    This decoder uses a U-Net architecture with multiple resolution levels
    to effectively extract hidden data across the whole image.
    
    Args:
        data_depth (int): Bits per pixel to extract
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__(data_depth, hidden_size)
        
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
        
        # Decoder path (upsampling)
        self.dec3 = self._make_decoder_block(hidden_size * 8, hidden_size * 4)
        self.dec2 = self._make_decoder_block(hidden_size * 4, hidden_size * 2)
        self.dec1 = self._make_decoder_block(hidden_size * 2, hidden_size)
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 2),
            nn.Conv2d(hidden_size // 2, data_depth, kernel_size=3, padding=1),
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
        
    def forward(self, stego_image):
        """
        Forward pass to extract hidden data from image.
        
        Args:
            stego_image (torch.Tensor): Steganographic image tensor (N, 3, H, W)
            
        Returns:
            torch.Tensor: Extracted message bits as logits (N, D, H, W)
        """
        # Encoder path
        enc1_out = self.enc1(stego_image)
        enc2_in = self.pool(enc1_out)
        
        enc2_out = self.enc2(enc2_in)
        enc3_in = self.pool(enc2_out)
        
        enc3_out = self.enc3(enc3_in)
        bottleneck_in = self.pool(enc3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(bottleneck_in)
        
        # Decoder path - no skip connections in decoder
        # Omitting skip connections forces the model to focus on the hidden data
        dec3_in = self.upsample(bottleneck_out)
        dec3_out = self.dec3(dec3_in)
        
        dec2_in = self.upsample(dec3_out)
        dec2_out = self.dec2(dec2_in)
        
        dec1_in = self.upsample(dec2_out)
        dec1_out = self.dec1(dec1_in)
        
        # Final extraction
        message = self.final(dec1_out)
        
        return message


class AttentionDecoder(BaseDecoder):
    """
    Attention-based decoder for robust extraction.
    
    This decoder uses self-attention to focus on image regions
    most likely to contain hidden data.
    
    Args:
        data_depth (int): Bits per pixel to extract
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, data_depth=1, hidden_size=64):
        super().__init__(data_depth, hidden_size)
        
        # Initial feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
        )
        
        # Self-attention module
        self.attention = SelfAttention(hidden_size * 2)
        
        # Message extraction
        self.extraction = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 2),
            
            # Output layer
            nn.Conv2d(hidden_size // 2, data_depth, kernel_size=3, padding=1),
        )
        
    def forward(self, stego_image):
        """
        Forward pass to extract hidden data from image.
        
        Args:
            stego_image (torch.Tensor): Steganographic image tensor (N, 3, H, W)
            
        Returns:
            torch.Tensor: Extracted message bits as logits (N, D, H, W)
        """
        # Feature extraction
        features = self.features(stego_image)
        
        # Apply attention
        attended_features = self.attention(features)
        
        # Extract message
        message = self.extraction(attended_features)
        
        return message


# Helper modules

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


class SelfAttention(nn.Module):
    """Self-attention module for focusing on relevant image regions."""
    
    def __init__(self, in_channels):
        super().__init__()
        # Query, key, value projections
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Gamma parameter for controlling attention strength
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # Create query, key, value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Calculate attention map
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=2)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        
        # Apply attention with residual connection
        out = self.gamma * out + x
        
        return out