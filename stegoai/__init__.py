# -*- coding: utf-8 -*-
"""
Stego-AI: Comprehensive steganography toolkit using deep learning.

This package provides tools for hiding and extracting data in various media
types including images, audio, text, video, and network protocols.
"""

__author__ = """Azhar"""
__email__ = 'skazharuddin2003@gmail.com'
__version__ = '0.2.1'

# Image steganography
from stegoai.models.image.models import ImageStegoNet

# Audio steganography
from stegoai.models.audio.models import AudioStegoNet

# Text steganography
from stegoai.models.text.models import TextStegoNet

# Video steganography
from stegoai.models.video.models import VideoStegoNet

# Network steganography
from stegoai.models.network.models import NetworkStegoNet

# Factory function to get the appropriate model based on media type
def get_model(media_type, **kwargs):
    """
    Factory function to get the appropriate steganography model.
    
    Args:
        media_type (str): Type of media ('image', 'audio', 'text', 'video', 'network')
        **kwargs: Additional arguments to pass to the model constructor
    
    Returns:
        A steganography model appropriate for the media type
    
    Raises:
        ValueError: If an invalid media type is provided
    """
    if media_type == 'image':
        # Handle architecture parameter specially for ImageStegoNet
        if 'architecture' in kwargs:
            architecture = kwargs.pop('architecture')
            return ImageStegoNet.load(architecture=architecture, **kwargs)
        else:
            return ImageStegoNet(**kwargs)
    elif media_type == 'audio':
        # Handle method parameter as architecture for AudioStegoNet
        if 'method' in kwargs:
            method = kwargs.pop('method')
            kwargs['mode'] = method  # Change method to mode parameter
            return AudioStegoNet.load(architecture=method, **kwargs)
        return AudioStegoNet(**kwargs)
    elif media_type == 'text':
        return TextStegoNet(**kwargs)
    elif media_type == 'video':
        return VideoStegoNet(**kwargs)
    elif media_type == 'network':
        return NetworkStegoNet(**kwargs)
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

__all__ = [
    'ImageStegoNet',
    'AudioStegoNet',
    'TextStegoNet',
    'VideoStegoNet',
    'NetworkStegoNet',
    'get_model',
]