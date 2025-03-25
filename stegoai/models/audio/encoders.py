# -*- coding: utf-8 -*-
"""
Encoder networks for audio steganography.

This module contains neural network architectures for encoding
hidden data into audio signals with minimal perceptual distortion.
"""

import torch
from torch import nn
import torch.nn.functional as F


class BaseAudioEncoder(nn.Module):
    """
    Base class for audio encoder networks.
    
    Args:
        data_depth (int): Bits per frame/sample to hide
        hidden_size (int): Size of hidden layers
        mode (str): Operating mode (spectrogram, waveform, phase)
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='spectrogram'):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.mode = mode


class SpectrogramEncoder(BaseAudioEncoder):
    """
    Encoder for hiding data in audio spectrograms.
    
    This encoder takes a magnitude spectrogram as input and
    produces a modified spectrogram containing hidden data.
    
    Input: (N, 1, F, T) - Batch of spectrograms
    Output: (N, 1, F, T) - Steganographic spectrograms
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='spectrogram'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Initial feature extraction
        self.features = nn.Sequential(
            # Input: (N, 1, F, T)
            nn.Conv2d(1, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Payload injection
        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_size + data_depth, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
            
            # Output layer (residual)
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
        )
        
    def forward(self, spectrogram, payload):
        """
        Forward pass to hide data in spectrogram.
        
        Args:
            spectrogram (torch.Tensor): Spectrogram tensor (N, 1, F, T)
            payload (torch.Tensor): Payload data tensor (N, D, F, T)
            
        Returns:
            torch.Tensor: Steganographic spectrogram (N, 1, F, T)
        """
        # Extract features
        features = self.features(spectrogram)
        
        # Concatenate features and payload
        combined = torch.cat([features, payload], dim=1)
        
        # Generate residual
        residual = self.encoder(combined)
        
        # Add residual to original spectrogram
        stego_spectrogram = spectrogram + residual
        
        return stego_spectrogram


class WaveformEncoder(BaseAudioEncoder):
    """
    Encoder for hiding data directly in audio waveforms.
    
    This encoder processes audio in the time domain and is suitable
    for applications where phase information must be preserved.
    
    Input: (N, T) - Batch of audio waveforms
    Output: (N, T) - Steganographic waveforms
    """
    
    def __init__(self, data_depth=1, hidden_size=64, mode='waveform'):
        super().__init__(data_depth, hidden_size, mode)
        
        # Use 1D convolutions for waveform processing
        self.features = nn.Sequential(
            # Input: (N, 1, T)
            nn.Conv1d(1, hidden_size, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size),
            
            nn.Conv1d(hidden_size, hidden_size, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size),
        )
        
        # Payload injection
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_size + data_depth, hidden_size * 2, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size * 2),
            
            nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=9, padding=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size),
            
            # Output layer (residual)
            nn.Conv1d(hidden_size, 1, kernel_size=9, padding=4),
            nn.Tanh(),  # Bound output to [-1, 1]
        )
        
    def forward(self, waveform, payload):
        """
        Forward pass to hide data in waveform.
        
        Args:
            waveform (torch.Tensor): Waveform tensor (N, T)
            payload (torch.Tensor): Payload data tensor (N, D, T)
            
        Returns:
            torch.Tensor: Steganographic waveform (N, T)
        """
        # Reshape waveform to (N, 1, T) for Conv1d
        x = waveform.unsqueeze(1)
        
        # Extract features
        features = self.features(x)
        
        # Ensure payload has the correct shape
        if features.shape[2] != payload.shape[2]:
            # Resize payload to match features
            if features.shape[2] < payload.shape[2]:
                payload = payload[:, :, :features.shape[2]]
            else:
                padding = torch.zeros(payload.shape[0], payload.shape[1], 
                                     features.shape[2] - payload.shape[2],
                                     device=payload.device)
                payload = torch.cat([payload, padding], dim=2)
        
        # Concatenate features and payload
        combined = torch.cat([features, payload], dim=1)
        
        # Generate output
        stego_waveform = self.encoder(combined)
        
        # Remove channel dimension
        stego_waveform = stego_waveform.squeeze(1)
        
        return stego_waveform


class PhaseEncoder(BaseAudioEncoder):
    """
    Encoder for hiding data in the phase component of spectrograms.
    
    This encoder modifies the phase component while preserving the
    magnitude, which can be less perceptible in some cases.
    
    Input: (N, 2, F, T) - Batch of complex spectrograms (magnitude and phase)
    Output: (N, 2, F, T) - Steganographic complex spectrograms
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
            nn.Conv2d(hidden_size + data_depth, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size * 2),
            
            nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(hidden_size),
        )
        
        # Separate outputs for magnitude and phase
        self.mag_out = nn.Sequential(
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
        )
        
        self.phase_out = nn.Sequential(
            nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
            nn.Tanh(),  # Bound phase modifications to [-1, 1]
        )
        
    def forward(self, spectrogram, payload):
        """
        Forward pass to hide data in complex spectrogram.
        
        Args:
            spectrogram (torch.Tensor): Complex spectrogram tensor (N, 2, F, T)
                                       where channel 0 is magnitude and channel 1 is phase
            payload (torch.Tensor): Payload data tensor (N, D, F, T)
            
        Returns:
            torch.Tensor: Steganographic complex spectrogram (N, 2, F, T)
        """
        # Split magnitude and phase
        magnitude = spectrogram[:, 0:1]  # Keep dim
        phase = spectrogram[:, 1:2]      # Keep dim
        
        # Extract features
        mag_features = self.mag_features(magnitude)
        phase_features = self.phase_features(phase)
        
        # Concatenate features
        features = torch.cat([mag_features, phase_features], dim=1)
        
        # Ensure payload has the correct shape
        if features.shape[2:] != payload.shape[2:]:
            # Resize payload to match features
            payload = F.interpolate(payload, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate features and payload
        combined = torch.cat([features, payload], dim=1)
        
        # Process combined features
        processed = self.combined(combined)
        
        # Generate separate outputs for magnitude and phase
        mag_residual = self.mag_out(processed)
        phase_residual = self.phase_out(processed)
        
        # Apply minimal changes to magnitude, focus on phase
        new_magnitude = magnitude + mag_residual * 0.1  # Small changes to magnitude
        new_phase = phase + phase_residual * torch.pi   # Larger changes to phase
        
        # Combine back into complex spectrogram
        stego_spectrogram = torch.cat([new_magnitude, new_phase], dim=1)
        
        return stego_spectrogram


class AttentionAudioEncoder(BaseAudioEncoder):
    """
    Advanced encoder with attention mechanism for audio steganography.
    
    This encoder uses attention mechanisms to identify optimal regions
    in the audio signal for hiding data with minimal perceptual impact.
    
    Can work in either spectrogram or waveform modes.
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
            )
            
            # Self-attention for waveform
            self.attention = WaveformAttention(hidden_size)
            
            # Encoder with payload injection
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden_size + data_depth, hidden_size * 2, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size * 2),
                
                nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=9, padding=4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size),
                
                # Output layer
                nn.Conv1d(hidden_size, 1, kernel_size=9, padding=4),
                nn.Tanh(),
            )
        else:
            # 2D convolutions for spectrogram
            self.features = nn.Sequential(
                nn.Conv2d(1, hidden_size, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
            )
            
            # Self-attention for spectrogram
            self.attention = SpectrogramAttention(hidden_size)
            
            # Encoder with payload injection
            self.encoder = nn.Sequential(
                nn.Conv2d(hidden_size + data_depth, hidden_size * 2, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size * 2),
                
                nn.Conv2d(hidden_size * 2, hidden_size, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
                
                # Output layer
                nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
            )
    
    def forward(self, audio, payload):
        """
        Forward pass to hide data in audio.
        
        Args:
            audio: Audio tensor (waveform or spectrogram)
            payload: Payload data tensor
            
        Returns:
            torch.Tensor: Steganographic audio
        """
        if self.mode == 'waveform':
            # Handle waveform (N, T)
            x = audio.unsqueeze(1)  # Add channel dim: (N, 1, T)
            
            # Extract features
            features = self.features(x)
            
            # Apply attention
            features = self.attention(features)
            
            # Resize payload if needed
            if features.shape[2] != payload.shape[2]:
                if features.shape[2] < payload.shape[2]:
                    payload = payload[:, :, :features.shape[2]]
                else:
                    padding = torch.zeros(payload.shape[0], payload.shape[1], 
                                         features.shape[2] - payload.shape[2],
                                         device=payload.device)
                    payload = torch.cat([payload, padding], dim=2)
            
            # Concatenate features and payload
            combined = torch.cat([features, payload], dim=1)
            
            # Generate output
            stego = self.encoder(combined)
            
            # Remove channel dimension
            stego = stego.squeeze(1)
            
        else:
            # Handle spectrogram (N, 1, F, T)
            if audio.dim() == 3:
                x = audio.unsqueeze(1)  # Add channel dim if needed
            else:
                x = audio
                
            # Extract features
            features = self.features(x)
            
            # Apply attention
            features = self.attention(features)
            
            # Resize payload if needed
            if features.shape[2:] != payload.shape[2:]:
                payload = F.interpolate(payload, size=features.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate features and payload
            combined = torch.cat([features, payload], dim=1)
            
            # Generate residual
            residual = self.encoder(combined)
            
            # Add residual to original
            stego = x + residual
            
        return stego


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
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x L
        
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
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, H, W)  # B x C x H x W
        
        # Residual connection with learnable weight
        out = self.gamma * out + x
        
        return out