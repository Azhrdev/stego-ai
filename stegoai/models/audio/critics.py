# -*- coding: utf-8 -*-
"""
Critic networks for audio steganography.

This module contains neural network architectures for detecting
steganographic data in audio signals, used during adversarial training.
"""

import torch
from torch import nn
import torch.nn.functional as F


class AudioCritic(nn.Module):
    """
    Critic network for detecting steganography in audio.
    
    This network attempts to distinguish between clean and
    steganographic audio, regardless of the domain.
    
    Args:
        hidden_size (int): Size of hidden layers
        mode (str): Operating mode (spectrogram, waveform, phase)
    """
    
    def __init__(self, hidden_size=64, mode='spectrogram'):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Different networks based on operating mode
        if mode == 'waveform':
            self.model = self._build_waveform_critic()
        else:  # spectrogram or phase
            self.model = self._build_spectrogram_critic()
    
    def _build_waveform_critic(self):
        """Build a critic network for waveform audio."""
        return nn.Sequential(
            # Input: (N, 1, T)
            nn.Conv1d(1, self.hidden_size, kernel_size=9, padding=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.hidden_size),
            
            nn.Conv1d(self.hidden_size, self.hidden_size * 2, kernel_size=9, padding=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.hidden_size * 2),
            
            nn.Conv1d(self.hidden_size * 2, self.hidden_size * 4, kernel_size=9, padding=4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.hidden_size * 4),
            
            nn.Conv1d(self.hidden_size * 4, self.hidden_size * 2, kernel_size=9, padding=4, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(self.hidden_size * 2),
            
            nn.Conv1d(self.hidden_size * 2, 1, kernel_size=9, padding=4, stride=1),
        )
    
    def _build_spectrogram_critic(self):
        """Build a critic network for spectrogram audio."""
        return nn.Sequential(
            # Input: (N, 1, F, T)
            nn.Conv2d(1, self.hidden_size, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            
            nn.Conv2d(self.hidden_size, self.hidden_size * 2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_size * 2),
            
            nn.Conv2d(self.hidden_size * 2, self.hidden_size * 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_size * 4),
            
            nn.Conv2d(self.hidden_size * 4, self.hidden_size * 2, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.hidden_size * 2),
            
            nn.Conv2d(self.hidden_size * 2, 1, kernel_size=3, padding=1, stride=1),
        )
    
    def forward(self, audio):
        """
        Forward pass to detect steganography.
        
        Args:
            audio: Audio tensor (waveform or spectrogram)
            
        Returns:
            torch.Tensor: Steganography detection score
        """
        # Add channel dimension if needed
        if self.mode == 'waveform' and audio.dim() == 2:
            x = audio.unsqueeze(1)  # (N, T) -> (N, 1, T)
        elif audio.dim() == 3 and self.mode != 'waveform':
            x = audio.unsqueeze(1)  # (N, F, T) -> (N, 1, F, T)
        else:
            x = audio
            
        # Process through the model
        x = self.model(x)
        
        # Global average pooling
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        
        return x


class SpectralAudioCritic(nn.Module):
    """
    Advanced critic that analyzes audio in both time and frequency domains.
    
    This critic uses spectral analysis techniques even for waveform inputs,
    leveraging frequency domain properties to detect subtle steganographic changes.
    
    Args:
        hidden_size (int): Size of hidden layers
        mode (str): Operating mode (spectrogram, waveform, phase)
    """
    
    def __init__(self, hidden_size=64, mode='spectrogram'):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Time domain analysis
        if mode == 'waveform':
            self.time_features = nn.Sequential(
                nn.Conv1d(1, hidden_size, kernel_size=9, padding=4, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size),
                
                nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=9, padding=4, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size * 2),
            )
        else:
            self.time_features = nn.Sequential(
                nn.Conv2d(1, hidden_size, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size),
                
                nn.Conv2d(hidden_size, hidden_size * 2, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size * 2),
            )
        
        # Frequency domain analysis (for all modes)
        # This will analyze spectral statistics after FFT
        self.freq_features = nn.Sequential(
            nn.Conv1d(128, hidden_size, kernel_size=3, padding=1),  # 128 frequency bins
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size),
            
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_size * 2),
        )
        
        # Combined analysis
        if mode == 'waveform':
            # For waveform, need to handle 1D time features
            self.combined = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size * 2),
                
                nn.Linear(hidden_size * 2, 1),
            )
        else:
            # For spectrogram, 2D time features
            self.combined = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size * 2),
                
                nn.Linear(hidden_size * 2, 1),
            )
    
    def forward(self, audio):
        """
        Forward pass to detect steganography using multi-domain analysis.
        
        Args:
            audio: Audio tensor (waveform or spectrogram)
            
        Returns:
            torch.Tensor: Steganography detection score
        """
        # Add channel dimension if needed
        if self.mode == 'waveform' and audio.dim() == 2:
            x = audio.unsqueeze(1)  # (N, T) -> (N, 1, T)
        elif audio.dim() == 3 and self.mode != 'waveform':
            x = audio.unsqueeze(1)  # (N, F, T) -> (N, 1, F, T)
        else:
            x = audio
        
        # Time domain analysis
        time_feats = self.time_features(x)
        
        # Frequency domain analysis
        if self.mode == 'waveform':
            # For waveform, compute FFT
            batch_size, _, seq_len = x.shape
            # Take chunks of the signal for FFT
            n_chunks = 10  # Number of chunks to analyze
            chunk_len = seq_len // n_chunks
            
            freq_feats_list = []
            for i in range(n_chunks):
                # Extract chunk
                if i * chunk_len + chunk_len <= seq_len:
                    chunk = x[:, :, i * chunk_len:i * chunk_len + chunk_len]
                    # Compute FFT
                    fft = torch.fft.rfft(chunk.squeeze(1), dim=1)
                    # Get magnitudes and normalize
                    magnitudes = torch.abs(fft)
                    # Resize to 128 bins
                    if magnitudes.shape[1] > 128:
                        magnitudes = F.interpolate(magnitudes.unsqueeze(1), size=128, mode='linear').squeeze(1)
                    elif magnitudes.shape[1] < 128:
                        pad_size = 128 - magnitudes.shape[1]
                        magnitudes = F.pad(magnitudes, (0, pad_size))
                    # Add to list
                    freq_feats_list.append(magnitudes)
            
            # Combine frequency features across chunks
            freq_input = torch.stack(freq_feats_list, dim=2)  # (B, 128, n_chunks)
        else:
            # For spectrogram, already in frequency domain
            # Extract frequency statistics across time
            freq_input = torch.mean(x.squeeze(1), dim=2)  # Average across time
            # Ensure we have 128 frequency bins
            if freq_input.shape[1] > 128:
                freq_input = F.interpolate(freq_input.unsqueeze(1), size=128, mode='linear').squeeze(1)
            elif freq_input.shape[1] < 128:
                pad_size = 128 - freq_input.shape[1]
                freq_input = F.pad(freq_input, (0, pad_size))
            # Add time dimension for Conv1D
            freq_input = freq_input.unsqueeze(2).expand(-1, -1, 10)  # (B, 128, 10)
        
        # Process frequency features
        freq_feats = self.freq_features(freq_input)
        
        # Global pooling for both domains
        if self.mode == 'waveform':
            time_global = torch.mean(time_feats, dim=2)  # (B, hidden_size*2)
            freq_global = torch.mean(freq_feats, dim=2)  # (B, hidden_size*2)
        else:
            time_global = torch.mean(time_feats.view(time_feats.size(0), time_feats.size(1), -1), dim=2)  # (B, hidden_size*2)
            freq_global = torch.mean(freq_feats, dim=2)  # (B, hidden_size*2)
        
        # Concatenate global features
        combined_feats = torch.cat([time_global, freq_global], dim=1)
        
        # Final classification
        score = self.combined(combined_feats)
        
        return score.squeeze(1)


class MultiResolutionAudioCritic(nn.Module):
    """
    Critic that analyzes audio at multiple resolutions.
    
    This critic examines different time scales and frequency bands
    to detect steganography more effectively.
    
    Args:
        hidden_size (int): Size of hidden layers
        mode (str): Operating mode (spectrogram, waveform, phase)
    """
    
    def __init__(self, hidden_size=64, mode='spectrogram'):
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Different branches for multi-scale analysis
        if mode == 'waveform':
            # Short-term analysis (small kernel)
            self.branch1 = nn.Sequential(
                nn.Conv1d(1, hidden_size // 3, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
                
                nn.Conv1d(hidden_size // 3, hidden_size // 3, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
            )
            
            # Medium-term analysis (medium kernel)
            self.branch2 = nn.Sequential(
                nn.Conv1d(1, hidden_size // 3, kernel_size=9, padding=4, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
                
                nn.Conv1d(hidden_size // 3, hidden_size // 3, kernel_size=9, padding=4, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
            )
            
            # Long-term analysis (large kernel)
            self.branch3 = nn.Sequential(
                nn.Conv1d(1, hidden_size // 3, kernel_size=27, padding=13, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm1d(hidden_size // 3),
                
                nn.Conv1d(hidden_size // 3, hidden_size // 3, kernel_size=27, padding=13, stride=2),
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
                
                nn.Conv1d(hidden_size, 1, kernel_size=9, padding=4),
            )
        else:
            # For spectrogram, different receptive fields
            # Local patterns
            self.branch1 = nn.Sequential(
                nn.Conv2d(1, hidden_size // 3, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
                
                nn.Conv2d(hidden_size // 3, hidden_size // 3, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
            )
            
            # Mid-range patterns with dilation
            self.branch2 = nn.Sequential(
                nn.Conv2d(1, hidden_size // 3, kernel_size=3, padding=2, dilation=2, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
                
                nn.Conv2d(hidden_size // 3, hidden_size // 3, kernel_size=3, padding=2, dilation=2, stride=2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
            )
            
            # Global context with pooling
            self.branch3 = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(1, hidden_size // 3, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(hidden_size // 3),
                
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(hidden_size // 3, hidden_size // 3, kernel_size=1),
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
                
                nn.Conv2d(hidden_size, 1, kernel_size=3, padding=1),
            )
    
    def forward(self, audio):
        """
        Forward pass to detect steganography at multiple resolutions.
        
        Args:
            audio: Audio tensor (waveform or spectrogram)
            
        Returns:
            torch.Tensor: Steganography detection score
        """
        # Add channel dimension if needed
        if self.mode == 'waveform' and audio.dim() == 2:
            x = audio.unsqueeze(1)  # (N, T) -> (N, 1, T)
        elif audio.dim() == 3 and self.mode != 'waveform':
            x = audio.unsqueeze(1)  # (N, F, T) -> (N, 1, F, T)
        else:
            x = audio
            
        # Process through different branches
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        
        # Ensure all feature maps have the same size
        if self.mode == 'waveform':
            # For 1D, resize in time dimension
            min_len = min(feat1.size(2), feat2.size(2), feat3.size(2))
            feat1 = feat1[:, :, :min_len]
            feat2 = feat2[:, :, :min_len]
            feat3 = feat3[:, :, :min_len]
        else:
            # For 2D, resize in both dimensions
            min_h = min(feat1.size(2), feat2.size(2), feat3.size(2))
            min_w = min(feat1.size(3), feat2.size(3), feat3.size(3))
            feat1 = feat1[:, :, :min_h, :min_w]
            feat2 = feat2[:, :, :min_h, :min_w]
            feat3 = feat3[:, :, :min_h, :min_w]
        
        # Concatenate features
        features = torch.cat([feat1, feat2, feat3], dim=1)
        
        # Combined processing
        x = self.combined(features)
        
        # Global average pooling
        x = torch.mean(x.view(x.size(0), -1), dim=1)
        
        return x


# Define a simple factory function to get the appropriate critic
def get_audio_critic(critic_type='standard', hidden_size=64, mode='spectrogram'):
    """
    Factory function to get an audio critic network.
    
    Args:
        critic_type: Type of critic ('standard', 'spectral', 'multiresolution')
        hidden_size: Size of hidden layers
        mode: Operating mode ('spectrogram', 'waveform', 'phase')
        
    Returns:
        An audio critic network
    """
    if critic_type == 'spectral':
        return SpectralAudioCritic(hidden_size=hidden_size, mode=mode)
    elif critic_type == 'multiresolution':
        return MultiResolutionAudioCritic(hidden_size=hidden_size, mode=mode)
    else:  # standard
        return AudioCritic(hidden_size=hidden_size, mode=mode)