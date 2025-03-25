# -*- coding: utf-8 -*-
"""
Base classes for steganography models.

This module provides the base classes for all steganography models,
defining a common interface and shared functionality.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import numpy as np
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


class BaseStegoNet(ABC):
    """
    Base class for all steganography models.
    
    This abstract class defines the interface that all steganography models
    should implement, including encoding and decoding methods.
    
    Attributes:
        data_depth (int): Depth of data to hide per element
        device (torch.device): Device to use for computation
        verbose (bool): Whether to print verbose output
    """
    
    def __init__(
        self, 
        data_depth: int = 1, 
        encoder: Optional[Any] = None,
        decoder: Optional[Any] = None,
        critic: Optional[Any] = None,
        hidden_size: int = 64, 
        cuda: bool = True, 
        verbose: bool = False, 
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the base steganography model.
        
        Args:
            data_depth: Number of bits to hide per element
            encoder: Encoder network class or instance
            decoder: Decoder network class or instance
            critic: Critic network class or instance
            hidden_size: Size of hidden layers in networks
            cuda: Whether to use GPU acceleration
            verbose: Whether to print verbose output
            log_dir: Directory to save logs and checkpoints
        """
        self.data_depth = data_depth
        self.hidden_size = hidden_size
        self.verbose = verbose
        self.log_dir = log_dir
        
        # Setup device (GPU/CPU)
        self.set_device(cuda)
        
        # Training state and metrics
        self.fit_metrics = None
        self.history = []
        self.epochs = 0
        
        # Create log directory if specified
        if log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self.samples_path = os.path.join(self.log_dir, 'samples')
            os.makedirs(self.samples_path, exist_ok=True)
            
        if verbose:
            logger.info(f"Initialized {self.__class__.__name__} with {data_depth} data depth")
    
    def set_device(self, cuda: bool = True) -> None:
        """
        Set the device (CPU/GPU) for the models.
        
        Args:
            cuda: Whether to use CUDA if available
        """
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if self.verbose:
            device_type = "GPU" if self.cuda else "CPU"
            logger.info(f"Using {device_type} device")
    
    @abstractmethod
    def encode(self, *args, **kwargs) -> Any:
        """
        Hide data in a cover medium.
        
        This method should be implemented by subclasses to hide data
        in a specific type of cover medium.
        """
        pass
    
    @abstractmethod
    def decode(self, *args, **kwargs) -> Any:
        """
        Extract hidden data from a steganographic medium.
        
        This method should be implemented by subclasses to extract data
        from a specific type of steganographic medium.
        """
        pass
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Any:
        """
        Train the model on cover and payload data.
        
        This method should be implemented by subclasses to train the model
        on a specific type of data.
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> 'BaseStegoNet':
        """
        Load a model from disk or use a predefined architecture.
        
        This method should be implemented by subclasses to load a model
        for a specific type of data.
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the model
        torch.save(self, path)
        
        if self.verbose:
            logger.info(f"Model saved to {path}")


class BaseEncoder(torch.nn.Module, ABC):
    """Base class for all encoder networks."""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass to encode data."""
        pass


class BaseDecoder(torch.nn.Module, ABC):
    """Base class for all decoder networks."""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass to decode data."""
        pass


class BaseCritic(torch.nn.Module, ABC):
    """Base class for all critic networks."""
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass to detect steganography."""
        pass