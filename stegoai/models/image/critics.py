# -*- coding: utf-8 -*-
"""
Critic networks for image steganography.

This module contains neural network architectures for distinguishing
between clean images and steganographic images, used during adversarial training.
"""

import torch
from torch import nn
import torch.nn.functional as F


class Critic(nn.Module):
    """
    Standard critic network for detecting steganography.
    
    This network attempts to distinguish between clean and 
    steganographic images, producing a score that should be 
    higher for steganographic images.
    
    Args:
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Input: (N, 3, H, W)
            nn.Conv2d(3, hidden_size, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 4),
            
            nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, 1, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, image):
        """
        Forward pass to detect steganography.
        
        Args:
            image (torch.Tensor): Image tensor to analyze (N, 3, H, W)
            
        Returns:
            torch.Tensor: Steganography detection score (N, 1)
        """
        # Feature extraction
        features = self.features(image)
        
        # Global average pooling to get a single score
        out = torch.mean(features.view(features.size(0), -1), dim=1)
        
        return out


class AdvancedCritic(nn.Module):
    """
    Advanced critic with residual blocks and multi-scale analysis.
    
    This network uses multiple pathways to analyze the image at different
    scales, making it more effective at detecting sophisticated steganography.
    
    Args:
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # Multi-scale feature extraction
        # Branch 1: Standard convolutional path
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, hidden_size // 3, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 3),
            
            nn.Conv2d(hidden_size // 3, hidden_size // 3, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 3),
        )
        
        # Branch 2: Dilated convolutions for wider receptive field
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, hidden_size // 3, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 3),
            
            nn.Conv2d(hidden_size // 3, hidden_size // 3, kernel_size=3, stride=2, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 3),
        )
        
        # Branch 3: Focused on high-frequency components
        self.branch3 = nn.Sequential(
            # High-pass filter approximation
            nn.Conv2d(3, hidden_size // 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 3),
            
            nn.Conv2d(hidden_size // 3, hidden_size // 3, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 3),
        )
        
        # Initialize high-pass filter weights for branch3 first layer
        with torch.no_grad():
            # Simple edge detection kernel
            hpf = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)
            hpf = hpf.view(1, 1, 3, 3).repeat(hidden_size // 3, 3, 1, 1) / 9.0
            self.branch3[0].weight.copy_(hpf)
        
        # Combined processing after concatenation
        self.combined = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            ResidualBlock(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, 1, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, image):
        """
        Forward pass to detect steganography.
        
        Args:
            image (torch.Tensor): Image tensor to analyze (N, 3, H, W)
            
        Returns:
            torch.Tensor: Steganography detection score (N, 1)
        """
        # Process through each branch
        feat1 = self.branch1(image)
        feat2 = self.branch2(image)
        feat3 = self.branch3(image)
        
        # Calculate common output size for all branches
        min_h = min(feat1.size(2), feat2.size(2), feat3.size(2))
        min_w = min(feat1.size(3), feat2.size(3), feat3.size(3))
        
        # Resize features to common size if needed
        if feat1.size(2) != min_h or feat1.size(3) != min_w:
            feat1 = F.interpolate(feat1, size=(min_h, min_w), mode='bilinear', align_corners=False)
        if feat2.size(2) != min_h or feat2.size(3) != min_w:
            feat2 = F.interpolate(feat2, size=(min_h, min_w), mode='bilinear', align_corners=False)
        if feat3.size(2) != min_h or feat3.size(3) != min_w:
            feat3 = F.interpolate(feat3, size=(min_h, min_w), mode='bilinear', align_corners=False)
        
        # Concatenate features
        combined_features = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Process combined features
        features = self.combined(combined_features)
        
        # Global average pooling
        out = torch.mean(features.view(features.size(0), -1), dim=1)
        
        return out


class SpectralCritic(nn.Module):
    """
    Critic that analyzes images in frequency domain.
    
    This network applies the Discrete Cosine Transform (DCT) to identify
    artifacts in the frequency domain, which is where many steganographic
    algorithms leave traces.
    
    Args:
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # Initial feature extraction in spatial domain
        self.spatial_features = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Processing for spatial features
        self.spatial_path = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
        )
        
        # Processing for frequency features
        # Input will be DCT coefficients organized as feature maps
        self.frequency_path = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Conv2d(hidden_size * 4, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
        )
        
    def dct_transform(self, x, block_size=8):
        """
        Apply DCT transform to input tensor using block processing.
        
        Args:
            x (torch.Tensor): Input tensor (N, C, H, W)
            block_size (int): Size of DCT blocks (default: 8)
            
        Returns:
            torch.Tensor: DCT coefficients organized as feature maps
        """
        batch_size, channels, height, width = x.shape
        
        # Pad to multiple of block_size if needed
        pad_h = (block_size - height % block_size) % block_size
        pad_w = (block_size - width % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        # Updated dimensions after padding
        _, _, height, width = x.shape
        
        # Number of blocks
        num_blocks_h = height // block_size
        num_blocks_w = width // block_size
        
        # Reshape for block processing
        x = x.view(batch_size, channels, num_blocks_h, block_size, num_blocks_w, block_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        x = x.view(batch_size * channels * num_blocks_h * num_blocks_w, block_size, block_size)
        
        # Create DCT basis functions
        i = torch.arange(block_size, dtype=torch.float32, device=x.device)
        j = torch.arange(block_size, dtype=torch.float32, device=x.device)
        
        # Create meshgrid of indices
        ii, jj = torch.meshgrid(i, j, indexing='ij')
        
        # DCT-II basis
        dct_basis = []
        for u in range(block_size):
            for v in range(block_size):
                basis = torch.cos((2*ii+1)*u*torch.pi/(2*block_size)) * \
                        torch.cos((2*jj+1)*v*torch.pi/(2*block_size))
                
                # Normalization
                if u == 0:
                    basis *= torch.sqrt(torch.tensor(1.0 / block_size, device=x.device))
                else:
                    basis *= torch.sqrt(torch.tensor(2.0 / block_size, device=x.device))
                    
                if v == 0:
                    basis *= torch.sqrt(torch.tensor(1.0 / block_size, device=x.device))
                else:
                    basis *= torch.sqrt(torch.tensor(2.0 / block_size, device=x.device))
                
                dct_basis.append(basis.view(-1))
        
        # Stack basis functions
        dct_basis = torch.stack(dct_basis, dim=0)
        
        # Apply DCT transform to each block
        x_flat = x.view(-1, block_size * block_size)
        dct_coeffs = torch.matmul(x_flat, dct_basis.t())
        
        # Reshape to block form
        dct_coeffs = dct_coeffs.view(-1, block_size, block_size)
        
        # Reshape back to image form but with coefficients organized differently
        # Group similar frequencies together
        dct_coeffs = dct_coeffs.view(batch_size, channels, num_blocks_h, num_blocks_w, block_size, block_size)
        dct_coeffs = dct_coeffs.permute(0, 1, 4, 5, 2, 3).contiguous()
        dct_coeffs = dct_coeffs.view(batch_size, channels * block_size * block_size, num_blocks_h, num_blocks_w)
        
        # We need to reduce channels for computational efficiency
        # Group DCT coefficients into meaningful frequency bands
        coeff_groups = []
        
        # Low frequency (DC + low AC)
        low_freq = dct_coeffs[:, :channels*4, :, :]
        coeff_groups.append(low_freq)
        
        # Mid frequency
        mid_freq = dct_coeffs[:, channels*4:channels*16, :, :]
        coeff_groups.append(mid_freq)
        
        # High frequency
        high_freq = dct_coeffs[:, channels*16:, :, :]
        coeff_groups.append(high_freq)
        
        # Resize and concatenate
        resized_groups = []
        for group in coeff_groups:
            # Resize to match spatial dimensions of original image
            resized = F.interpolate(group, size=(height, width), mode='nearest')
            resized_groups.append(resized)
        
        # Concatenate and reduce channels
        dct_features = torch.cat(resized_groups, dim=1)
        
        return dct_features
    
    def forward(self, image):
        """
        Forward pass to detect steganography using both spatial and frequency analysis.
        
        Args:
            image (torch.Tensor): Image tensor to analyze (N, 3, H, W)
            
        Returns:
            torch.Tensor: Steganography detection score (N, 1)
        """
        # Extract initial features
        features = self.spatial_features(image)
        
        # Process in spatial domain
        spatial_feats = self.spatial_path(features)
        
        # Apply frequency transform (simplified DCT simulation)
        # Real DCT would be computationally intensive, so we simulate with convolutions
        freq_kernel = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=torch.float32, device=image.device).view(1, 1, 3, 3).repeat(hidden_size, 1, 1, 1)
        
        freq_features = F.conv2d(
            image.repeat(1, hidden_size, 1, 1), 
            freq_kernel, 
            groups=hidden_size,
            padding=1
        )
        
        # Process frequency features
        freq_feats = self.frequency_path(freq_features)
        
        # Ensure same spatial dimensions
        if spatial_feats.size(2) != freq_feats.size(2) or spatial_feats.size(3) != freq_feats.size(3):
            # Resize to smaller dimension
            if spatial_feats.size(2) * spatial_feats.size(3) < freq_feats.size(2) * freq_feats.size(3):
                freq_feats = F.interpolate(
                    freq_feats, 
                    size=(spatial_feats.size(2), spatial_feats.size(3)), 
                    mode='bilinear',
                    align_corners=False
                )
            else:
                spatial_feats = F.interpolate(
                    spatial_feats, 
                    size=(freq_feats.size(2), freq_feats.size(3)), 
                    mode='bilinear',
                    align_corners=False
                )
        
        # Combine features
        combined_feats = torch.cat([spatial_feats, freq_feats], dim=1)
        
        # Final processing
        out = self.combined(combined_feats)
        
        # Global average pooling
        out = torch.mean(out.view(out.size(0), -1), dim=1)
        
        return out


class EnsembleCritic(nn.Module):
    """
    Ensemble of critics for more robust detection.
    
    This network combines multiple critic architectures to improve
    detection across different steganographic techniques.
    
    Args:
        hidden_size (int): Size of hidden layers
    """
    
    def __init__(self, hidden_size=64):
        super().__init__()
        
        # Create ensemble of critics
        self.critics = nn.ModuleList([
            Critic(hidden_size),
            AdvancedCritic(hidden_size),
            # Simplified version of SpectralCritic for efficiency
            nn.Sequential(
                nn.Conv2d(3, hidden_size, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
                
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size * 2),
                
                nn.Conv2d(hidden_size * 2, hidden_size * 4, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size * 4),
                
                nn.Conv2d(hidden_size * 4, 1, kernel_size=3, stride=1, padding=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        ])
        
        # Fusion layer to combine critic outputs
        self.fusion = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
        )
        
    def forward(self, image):
        """
        Forward pass using ensemble of critics.
        
        Args:
            image (torch.Tensor): Image tensor to analyze (N, 3, H, W)
            
        Returns:
            torch.Tensor: Steganography detection score (N, 1)
        """
        # Get predictions from each critic
        critic_outputs = []
        for critic in self.critics:
            output = critic(image)
            critic_outputs.append(output.view(-1, 1))
        
        # Combine outputs
        combined = torch.cat(critic_outputs, dim=1)
        
        # Apply fusion layer
        final_score = self.fusion(combined)
        
        return final_score.view(-1)


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