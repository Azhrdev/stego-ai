# -*- coding: utf-8 -*-
"""
Decoder networks for audio steganography.

This module contains neural network architectures for extracting
hidden data from steganographic audio signals.
"""

import torch
from torch import nn
import torch.nn.functional as F


class BaseAudioDecoder(nn.Module):
    """
    Base class for audio decoder networks.
    
    Args:
        data_depth (int): Bits per frame/sample to extract
        hidden_size (int): Size of hidden layers
        mode (str): Operating mode (spectrogram, waveform, phase)
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='spectrogram'):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.mode = mode


class SpectrogramDecoder(BaseAudioDecoder):
    """
    Decoder for extracting data from audio spectrograms.
    
    This decoder takes a steganographic spectrogram as input and
    extracts the hidden data.
    
    Input: (N, 1, F, T) - Steganographic spectrograms
    Output: (N, D, F, T) - Extracted payload data
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='spectrogram'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Input: (N, 1, F, T)
            nn.Conv2d(1, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
        )
        
        # Payload extraction
        self.extractor = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            # Output layer (no activation - using sigmoid in loss function)
            nn.Conv2d(hidden_size, data_depth, kernel_size=3, padding=1),
        )
        
    def forward(self, spectrogram):
        """
        Forward pass to extract data from spectrogram.
        
        Args:
            spectrogram (torch.Tensor): Steganographic spectrogram tensor (N, 1, F, T)
                                       or (N, F, T) which will be reshaped
            
        Returns:
            torch.Tensor: Extracted payload data (N, D, F, T)
        """
        # Ensure input has channel dimension
        if spectrogram.dim() == 3:
            x = spectrogram.unsqueeze(1)
        else:
            x = spectrogram
            
        # Extract features
        features = self.features(x)
        
        # Extract payload
        payload = self.extractor(features)
        
        return payload


class WaveformDecoder(BaseAudioDecoder):
    """
    Decoder for extracting data directly from audio waveforms.
    
    This decoder processes audio in the time domain to extract
    hidden data.
    
    Input: (N, T) - Steganographic waveforms
    Output: (N, D, T) - Extracted payload data
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='waveform'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Use 1D convolutions for waveform processing
        self.features = nn.Sequential(
            # Input: (N, 1, T)
            nn.Conv1d(1, hidden_size, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size),
            
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size * 2),
            
            nn.Conv1d(hidden_size * 2, hidden_size * 2, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size * 2),
        )
        
        # Payload extraction
        self.extractor = nn.Sequential(
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size),
            
            # Output layer (no activation - using sigmoid in loss function)
            nn.Conv1d(hidden_size, data_depth, kernel_size=9, padding=4),
        )
        
    def forward(self, waveform):
        """
        Forward pass to extract data from waveform.
        
        Args:
            waveform (torch.Tensor): Steganographic waveform tensor (N, T)
            
        Returns:
            torch.Tensor: Extracted payload data (N, D, T)
        """
        # Add channel dimension
        x = waveform.unsqueeze(1)
        
        # Extract features
        features = self.features(x)
        
        # Extract payload
        payload = self.extractor(features)
        
        return payload


class PhaseDecoder(BaseAudioDecoder):
    """
    Decoder for extracting data from the phase component of spectrograms.
    
    This decoder focuses on examining phase modifications to extract
    hidden data.
    
    Input: (N, 2, F, T) - Steganographic complex spectrograms
    Output: (N, D, F, T) - Extracted payload data
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='phase'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Process magnitude and phase separately
        self.mag_features = nn.Sequential(
            nn.Conv2d(1, hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 2),
        )
        
        self.phase_features = nn.Sequential(
            nn.Conv2d(1, hidden_size // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size // 2),
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
        )
        
        # Payload extraction
        self.extractor = nn.Sequential(
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            # Output layer (no activation - using sigmoid in loss function)
            nn.Conv2d(hidden_size, data_depth, kernel_size=3, padding=1),
        )
        
    def forward(self, spectrogram):
        """
        Forward pass to extract data from complex spectrogram.
        
        Args:
            spectrogram (torch.Tensor): Complex spectrogram tensor (N, 2, F, T)
                                       where channel 0 is magnitude and channel 1 is phase
            
        Returns:
            torch.Tensor: Extracted payload data (N, D, F, T)
        """
        # Split magnitude and phase
        magnitude = spectrogram[:, 0:1]  # Keep dim
        phase = spectrogram[:, 1:2]      # Keep dim
        
        # Extract features
        mag_features = self.mag_features(magnitude)
        phase_features = self.phase_features(phase)
        
        # Concatenate features
        features = torch.cat([mag_features, phase_features], dim=1)
        
        # Process combined features
        processed = self.combined(features)
        
        # Extract payload
        payload = self.extractor(processed)
        
        return payload


class MultiResolutionDecoder(BaseAudioDecoder):
    """
    Advanced decoder that analyzes audio at multiple resolutions.
    
    This decoder can extract data more robustly by examining the audio
    at different time scales.
    
    Works with both spectrograms and waveforms.
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='spectrogram'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Multi-resolution approach for either domain
        self.mode = mode
        
        if mode == 'waveform':
            # For waveform, use different kernel sizes
            self.branch1 = nn.Sequential(
                nn.Conv1d(1, hidden_size // 3, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
            )
            
            self.branch2 = nn.Sequential(
                nn.Conv1d(1, hidden_size // 3, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
            )
            
            self.branch3 = nn.Sequential(
                nn.Conv1d(1, hidden_size // 3, kernel_size=27, padding=13),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
            )
            
            # Combined processing
            self.combined = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size * 2),
                
                nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size),
            )
            
            # Output layer
            self.extractor = nn.Conv1d(hidden_size, data_depth, kernel_size=9, padding=4)
            
        else:
            # For spectrogram, use different types of processing
            self.branch1 = nn.Sequential(
                nn.Conv2d(1, hidden_size // 3, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
            )
            
            # Branch with dilation for wider context
            self.branch2 = nn.Sequential(
                nn.Conv2d(1, hidden_size // 3, kernel_size=3, padding=2, dilation=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
            )
            
            # Branch with pooling for global context
            self.branch3 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1, hidden_size // 3, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
            )
            
            # Combined processing
            self.combined = nn.Sequential(
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size * 2),
                
                nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
            )
            
            # Output layer
            self.extractor = nn.Conv2d(hidden_size, data_depth, kernel_size=3, padding=1)
    
    def forward(self, audio):
        """
        Forward pass to extract data from audio.
        
        Args:
            audio: Audio tensor (waveform or spectrogram)
            
        Returns:
            torch.Tensor: Extracted payload data
        """
        # Add channel dimension if needed
        if audio.dim() == 2 and self.mode == 'waveform':
            x = audio.unsqueeze(1)  # (N, T) -> (N, 1, T)
        elif audio.dim() == 3 and self.mode == 'spectrogram':
            x = audio.unsqueeze(1)  # (N, F, T) -> (N, 1, F, T)
        else:
            x = audio
            
        # Process through different branches
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        # Concatenate features
        features = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Combined processing
        processed = self.combined(features)
        
        # Extract payload
        payload = self.extractor(processed)
        
        return payload


class AttentionAudioDecoder(BaseAudioDecoder):
    """
    Decoder with attention mechanism for audio steganography.
    
    This decoder uses attention to focus on important regions of the audio
    that are likely to contain hidden data.
    
    Works with both spectrograms and waveforms.
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='spectrogram'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Different initialization based on mode
        if mode == 'waveform':
            # 1D convolutions for waveform
            self.features = nn.Sequential(
                nn.Conv1d(1, hidden_size, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size),
                
                nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size * 2),
            )
            
            # Self-attention for waveform
            self.attention = WaveformAttention(hidden_size * 2)
            
            # Extractor
            self.extractor = nn.Sequential(
                nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size),
                
                # Output layer
                nn.Conv1d(hidden_size, data_depth, kernel_size=9, padding=4),
            )
        else:
            # 2D convolutions for spectrogram
            self.features = nn.Sequential(
                nn.Conv2d(1, hidden_size, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
                
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size * 2),
            )
            
            # Self-attention for spectrogram
            self.attention = SpectrogramAttention(hidden_size * 2)
            
            # Extractor
            self.extractor = nn.Sequential(
                nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
                
                # Output layer
                nn.Conv2d(hidden_size, data_depth, kernel_size=3, padding=1),
            )
    
    def forward(self, audio):
        """
        Forward pass to extract data from audio.
        
        Args:
            audio: Audio tensor (waveform or spectrogram)
            
        Returns:
            torch.Tensor: Extracted payload data
        """
        if self.mode == 'waveform':
            # Handle waveform (N, T)
            if audio.dim() == 2:
                x = audio.unsqueeze(1)  # Add channel dim: (N, 1, T)
            else:
                x = audio
                
            # Extract features
            features = self.features(x)
            
            # Apply attention
            att_features = self.attention(features)
            
            # Extract payload
            payload = self.extractor(att_features)
            
        else:
            # Handle spectrogram (N, 1, F, T) or (N, F, T)
            if audio.dim() == 3:
                x = audio.unsqueeze(1)  # Add channel dim if needed
            else:
                x = audio
                
            # Extract features
            features = self.features(x)
            
            # Apply attention
            att_features = self.attention(features)
            
            # Extract payload
            payload = self.extractor(att_features)
            
        return payload


# Helper modules

class WaveformAttention(nn.Module):
    """Self-attention module for waveform data."""
    
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, L = x.size()
        
        # Project to get query, key, value
        proj_query = self.query(x).permute(0, 2, 1)  # B x L x C'
        proj_key = self.key(x)  # B x C' x L
        energy = torch.bmm(proj_query, proj_key)  # B x L x L
        attention = F.softmax(energy, dim=2)
        
        proj_value = self.value(x)  # B x C x L
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, H, W)  # B x C x H x W
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class SpectrogramAttention(nn.Module):
    """Self-attention module for spectrogram data."""
    
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Flatten spatial dimensions
        proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)  # B x HW x C'
        proj_key = self.key(x).view(batch_size, -1, H*W)  # B x C' x HW
        energy = torch.bmm(proj_query, proj_key)  # B x HW x HW
        attention = F.softmax(energy, dim=2)
        
        proj_value = self.value(x).view(batch_size, -1, H*W)  # B x C x HW
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x