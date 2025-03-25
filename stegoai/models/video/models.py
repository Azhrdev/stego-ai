# -*- coding: utf-8 -*-
"""
Video steganography models for Stego-AI.

This module contains the main VideoStegoNet class for video steganography,
implementing various algorithms for hiding data in video frames.
"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import cv2
import torch
from tqdm import tqdm

from stegoai.models.base import BaseStegoNet
from stegoai.models.image.models import ImageStegoNet
from stegoai.utils.text_utils import text_to_bits, bits_to_bytearray, bytearray_to_text

# Set up logging
logger = logging.getLogger(__name__)


class VideoStegoNet(BaseStegoNet):
    """
    Video steganography model.
    
    This class implements various algorithms for hiding data in videos:
    - frame_lsb: Modifies the least significant bits of select frames
    - frame_dct: Hides data in DCT coefficients like JPEG
    - temporal: Uses patterns across frames
    - neural: Applies image steganography to key frames
    
    Attributes:
        method (str): Steganography method to use
        image_model (ImageStegoNet): Image steganography model for neural method
    """
    
    METHODS = {
        'frame_lsb': {
            'capacity': 'high',
            'robustness': 'low',
            'visibility': 'low',
        },
        'frame_dct': {
            'capacity': 'medium',
            'robustness': 'medium',
            'visibility': 'low',
        },
        'temporal': {
            'capacity': 'low',
            'robustness': 'high',
            'visibility': 'medium',
        },
        'neural': {
            'capacity': 'high',
            'robustness': 'high',
            'visibility': 'medium',
        },
    }
    
    def __init__(
        self,
        method: str = 'frame_lsb',
        data_depth: int = 1,
        image_model: Optional[Any] = None,
        cuda: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize VideoStegoNet.
        
        Args:
            method: Steganography method (default: 'frame_lsb')
            data_depth: Bits per pixel to hide (default: 1)
            image_model: Optional ImageStegoNet for neural method
            cuda: Whether to use GPU if available
            verbose: Whether to print verbose output
        """
        super().__init__(
            data_depth=data_depth,
            cuda=cuda,
            verbose=verbose,
        )
        
        self.method = method
        
        # Validate method
        if self.method not in self.METHODS:
            raise ValueError(f"Unsupported method: {method}. Supported: {list(self.METHODS.keys())}")
        
        # For neural method, initialize image model
        self.image_model = None
        if self.method == 'neural':
            if image_model is not None:
                self.image_model = image_model
            else:
                # Create a default image model
                self.image_model = ImageStegoNet(
                    data_depth=data_depth,
                    architecture='dense',
                    cuda=cuda,
                    verbose=verbose
                )
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available."""
        try:
            subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Some functionality may be limited.")
            return False
    
    def _get_video_info(self, video_path: str) -> Dict:
        """
        Get video information using FFmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video information
        """
        try:
            # Check if FFmpeg is available
            if not self._check_ffmpeg():
                # Fall back to OpenCV
                return self._get_video_info_cv2(video_path)
            
            # Use FFmpeg to get video info
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-of', 'csv=p=0',
                video_path
            ]
            
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip().split(',')
            
            width, height = int(output[0]), int(output[1])
            
            # Parse frame rate fraction
            fps_parts = output[2].split('/')
            if len(fps_parts) == 2:
                fps = float(fps_parts[0]) / float(fps_parts[1])
            else:
                fps = float(fps_parts[0])
            
            duration = float(output[3]) if len(output) > 3 else None
            
            # Calculate total frames
            total_frames = int(duration * fps) if duration else None
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
            }
        
        except Exception as e:
            logger.error(f"Error getting video info with FFmpeg: {e}")
            # Fall back to OpenCV
            return self._get_video_info_cv2(video_path)
    
    def _get_video_info_cv2(self, video_path: str) -> Dict:
        """
        Get video information using OpenCV.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video information
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else None
            
            cap.release()
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
            }
        
        except Exception as e:
            logger.error(f"Error getting video info with OpenCV: {e}")
            raise
    
    def _extract_frames(self, video_path: str, output_dir: str, frame_indices: Optional[List[int]] = None) -> List[str]:
        """
        Extract specific frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            frame_indices: Indices of frames to extract (None for all frames)
            
        Returns:
            list: Paths to extracted frame images
        """
        try:
            # Make sure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Get video info
            video_info = self._get_video_info(video_path)
            total_frames = video_info['total_frames']
            
            # Validate frame indices
            if frame_indices is not None:
                if max(frame_indices) >= total_frames:
                    raise ValueError(f"Frame index {max(frame_indices)} exceeds total frames {total_frames}")
            
            # Extract frames using OpenCV
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            frame_paths = []
            current_frame = 0
            
            with tqdm(total=len(frame_indices) if frame_indices else total_frames, 
                    disable=not self.verbose, desc="Extracting frames") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_indices is None or current_frame in frame_indices:
                        frame_path = os.path.join(output_dir, f"frame_{current_frame:06d}.png")
                        cv2.imwrite(frame_path, frame)
                        frame_paths.append(frame_path)
                        pbar.update(1)
                    
                    current_frame += 1
            
            cap.release()
            
            return frame_paths
        
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            raise
    
    def _create_video_from_frames(self, frame_paths: List[str], output_path: str, 
                                fps: float, width: int = None, height: int = None,
                                quality: int = 23, codec: str = 'libx264') -> str:
        """
        Create video from frames.
        
        Args:
            frame_paths: Paths to frame images
            output_path: Path to output video
            fps: Frames per second
            width: Video width (None to use frame width)
            height: Video height (None to use frame height)
            quality: Video quality (CRF value, lower is better)
            codec: Video codec (default: 'libx264')
            
        Returns:
            str: Path to created video
        """
        try:
            # Check if frames exist
            if not frame_paths:
                raise ValueError("No frames provided")
            
            # Get frame dimensions if not provided
            if width is None or height is None:
                frame = cv2.imread(frame_paths[0])
                if frame is None:
                    raise ValueError(f"Could not read frame: {frame_paths[0]}")
                height, width = frame.shape[:2]
            
            # Check if FFmpeg is available
            if self._check_ffmpeg():
                # Use FFmpeg to create video
                cmd = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-r', str(fps),  # Frame rate
                    '-f', 'image2',  # Format
                    '-s', f"{width}x{height}",  # Size
                    '-i', os.path.join(os.path.dirname(frame_paths[0]), "frame_%06d.png"),  # Input pattern
                    '-c:v', codec,  # Codec
                    '-crf', str(quality),  # Quality
                    '-pix_fmt', 'yuv420p',  # Pixel format
                    output_path
                ]
                
                subprocess.check_call(cmd, stderr=subprocess.STDOUT)
            else:
                # Use OpenCV to create video
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                for frame_path in tqdm(frame_paths, disable=not self.verbose, desc="Creating video"):
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        logger.warning(f"Could not read frame: {frame_path}")
                        continue
                    
                    out.write(frame)
                
                out.release()
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating video: {e}")
            raise
    
    def _select_frame_indices(self, total_frames: int, message_bits: List[int]) -> List[int]:
        """
        Select frame indices for hiding data.
        
        Args:
            total_frames: Total number of frames
            message_bits: Message bits to hide
            
        Returns:
            list: Selected frame indices
        """
        # Calculate frames needed based on method
        if self.method == 'frame_lsb':
            # Each frame can store (width * height * 3 * data_depth) bits
            # For simplicity, we'll just use evenly spaced frames
            frames_needed = len(message_bits) // (1000 * self.data_depth) + 1
        elif self.method == 'frame_dct':
            # DCT has lower capacity than LSB
            frames_needed = len(message_bits) // (500 * self.data_depth) + 1
        elif self.method == 'temporal':
            # Temporal requires using consecutive frames
            frames_needed = len(message_bits) + 10  # Extra frames for robustness
        elif self.method == 'neural':
            # Neural method can store more data per frame
            frames_needed = len(message_bits) // (5000 * self.data_depth) + 1
        else:
            frames_needed = len(message_bits) // 1000 + 1
        
        # Ensure we don't exceed total frames
        frames_needed = min(frames_needed, total_frames)
        
        # For temporal method, use consecutive frames
        if self.method == 'temporal':
            # Start from a random position
            start_idx = random.randint(0, total_frames - frames_needed)
            return list(range(start_idx, start_idx + frames_needed))
        
        # For other methods, use evenly spaced frames
        step = max(1, total_frames // frames_needed)
        selected_indices = [i for i in range(0, total_frames, step)]
        
        # Limit to frames needed
        return selected_indices[:frames_needed]
    
    def encode(self, cover_path: str, output_path: str, message: str, quality: int = 23) -> None:
        """
        Hide a message in a video.
        
        Args:
            cover_path: Path to cover video
            output_path: Path for output steganographic video
            message: Message to hide
            quality: Video quality (CRF value, lower is better)
        """
        try:
            # Convert message to bits
            message_bits = text_to_bits(message)
            
            # Add termination marker (32 zeros)
            message_bits = message_bits + [0] * 32
            
            # Get video info
            video_info = self._get_video_info(cover_path)
            total_frames = video_info['total_frames']
            
            # Ensure we have enough frames
            if total_frames < 10:
                raise ValueError(f"Video has too few frames ({total_frames})")
            
            # Select frames for data hiding
            frame_indices = self._select_frame_indices(total_frames, message_bits)
            
            if self.verbose:
                logger.info(f"Using {len(frame_indices)} frames out of {total_frames} total frames")
            
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract selected frames
                frame_paths = self._extract_frames(cover_path, temp_dir, frame_indices)
                
                # Apply steganography method
                if self.method == 'frame_lsb':
                    self._encode_frame_lsb(frame_paths, message_bits)
                elif self.method == 'frame_dct':
                    self._encode_frame_dct(frame_paths, message_bits)
                elif self.method == 'temporal':
                    self._encode_temporal(frame_paths, message_bits)
                elif self.method == 'neural':
                    self._encode_neural(frame_paths, message_bits)
                else:
                    raise ValueError(f"Unknown steganography method: {self.method}")
                
                # Create output video from modified frames
                self._create_video_from_frames(
                    frame_paths, output_path, 
                    fps=video_info['fps'],
                    width=video_info['width'],
                    height=video_info['height'],
                    quality=quality
                )
            
            if self.verbose:
                logger.info(f"Message encoded successfully in {output_path}")
        
        except Exception as e:
            logger.error(f"Error in video encoding: {e}")
            raise
    
    def decode(self, stego_path: str) -> str:
        """
        Extract a hidden message from a video.
        
        Args:
            stego_path: Path to steganographic video
            
        Returns:
            str: Extracted message
        """
        try:
            # Get video info
            video_info = self._get_video_info(stego_path)
            total_frames = video_info['total_frames']
            
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract all frames (we don't know which ones contain data)
                # For efficiency, we'll use the same frame selection as encoding
                # but this assumes we know how the data was hidden
                
                # For demonstration, we'll extract frames based on method
                if self.method == 'temporal':
                    # For temporal, we need consecutive frames
                    # Extract all frames for simplicity
                    frame_paths = self._extract_frames(stego_path, temp_dir)
                else:
                    # For other methods, we can use frame selection heuristics
                    # This is a simplification - in reality, we would need to
                    # detect which frames contain data
                    
                    # Use a heuristic based on method
                    if self.method == 'frame_lsb':
                        step = max(1, total_frames // 50)
                    elif self.method == 'frame_dct':
                        step = max(1, total_frames // 30)
                    elif self.method == 'neural':
                        step = max(1, total_frames // 10)
                    else:
                        step = max(1, total_frames // 20)
                    
                    frame_indices = list(range(0, total_frames, step))
                    frame_paths = self._extract_frames(stego_path, temp_dir, frame_indices)
                
                # Apply steganography method for decoding
                if self.method == 'frame_lsb':
                    bits = self._decode_frame_lsb(frame_paths)
                elif self.method == 'frame_dct':
                    bits = self._decode_frame_dct(frame_paths)
                elif self.method == 'temporal':
                    bits = self._decode_temporal(frame_paths)
                elif self.method == 'neural':
                    bits = self._decode_neural(frame_paths)
                else:
                    raise ValueError(f"Unknown steganography method: {self.method}")
                
                # Find termination sequence
                term_index = -1
                for i in range(len(bits) - 31):
                    if all(bit == 0 for bit in bits[i:i+32]):
                        term_index = i
                        break
                
                # Extract message bits (up to termination if found)
                if term_index != -1:
                    message_bits = bits[:term_index]
                else:
                    message_bits = bits
                
                # Convert bits to text
                if not message_bits:
                    raise ValueError("No hidden message found in the video")
                
                byte_data = bits_to_bytearray(message_bits)
                message = bytearray_to_text(byte_data)
                
                if self.verbose:
                    logger.info(f"Message decoded successfully ({len(message)} characters)")
                
                return message
        
        except Exception as e:
            logger.error(f"Error in video decoding: {e}")
            raise
    
    def _encode_frame_lsb(self, frame_paths: List[str], message_bits: List[int]) -> None:
        """
        Hide message using LSB method on frames.
        
        Args:
            frame_paths: Paths to frame images
            message_bits: Message bits to hide
        """
        import random
        
        # Initialize bit index
        bit_index = 0
        
        # Process each frame until all bits are hidden
        for frame_path in tqdm(frame_paths, disable=not self.verbose, desc="Encoding frames"):
            if bit_index >= len(message_bits):
                break
            
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Calculate available capacity for this frame
            height, width, channels = frame.shape
            capacity = height * width * channels * self.data_depth
            
            # Determine how many bits to hide in this frame
            bits_to_hide = min(capacity, len(message_bits) - bit_index)
            
            # Generate random pixel indices
            pixel_indices = []
            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        pixel_indices.append((y, x, c))
            
            # Shuffle indices for better security
            random.shuffle(pixel_indices)
            
            # Hide bits
            for i in range(bits_to_hide):
                if bit_index >= len(message_bits):
                    break
                
                y, x, c = pixel_indices[i % len(pixel_indices)]
                
                # Get current pixel value
                pixel_value = frame[y, x, c]
                
                # Clear LSB and set to message bit
                frame[y, x, c] = (pixel_value & ~1) | message_bits[bit_index]
                
                bit_index += 1
            
            # Save modified frame
            cv2.imwrite(frame_path, frame)
    
    def _decode_frame_lsb(self, frame_paths: List[str]) -> List[int]:
        """
        Extract message from LSB method on frames.
        
        Args:
            frame_paths: Paths to frame images
            
        Returns:
            list: Extracted message bits
        """
        import random
        
        # Initialize extracted bits
        extracted_bits = []
        
        # Process each frame
        for frame_path in tqdm(frame_paths, disable=not self.verbose, desc="Decoding frames"):
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Calculate available capacity for this frame
            height, width, channels = frame.shape
            capacity = height * width * channels * self.data_depth
            
            # Generate random pixel indices (must match encoding order)
            pixel_indices = []
            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        pixel_indices.append((y, x, c))
            
            # Shuffle indices (must use same seed as encoding)
            random.shuffle(pixel_indices)
            
            # Extract bits
            for i in range(capacity):
                y, x, c = pixel_indices[i % len(pixel_indices)]
                
                # Get current pixel value
                pixel_value = frame[y, x, c]
                
                # Extract LSB
                bit = pixel_value & 1
                extracted_bits.append(bit)
                
                # Check for termination sequence
                if len(extracted_bits) >= 32 and all(bit == 0 for bit in extracted_bits[-32:]):
                    return extracted_bits[:-32]  # Remove termination sequence
        
        return extracted_bits
    
    def _encode_frame_dct(self, frame_paths: List[str], message_bits: List[int]) -> None:
        """
        Hide message using DCT coefficients method on frames.
        
        Args:
            frame_paths: Paths to frame images
            message_bits: Message bits to hide
        """
        # Initialize bit index
        bit_index = 0
        
        # Process each frame until all bits are hidden
        for frame_path in tqdm(frame_paths, disable=not self.verbose, desc="Encoding frames"):
            if bit_index >= len(message_bits):
                break
            
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Convert to YCrCb (separate luminance from chrominance)
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            # Split channels
            y, cr, cb = cv2.split(ycrcb)
            
            # Process 8x8 blocks in the Y channel
            height, width = y.shape
            bits_hidden_in_frame = 0
            
            for i in range(0, height - 8, 8):
                for j in range(0, width - 8, 8):
                    if bit_index >= len(message_bits):
                        break
                    
                    # Extract 8x8 block
                    block = y[i:i+8, j:j+8].astype(np.float32)
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Hide bit in the mid-frequency coefficient (4,3)
                    # Choose a coefficient that balances capacity and visibility
                    if message_bits[bit_index] == 1:
                        # Ensure coefficient is positive and above threshold
                        if dct_block[4, 3] < 3:
                            dct_block[4, 3] = 3
                    else:
                        # Ensure coefficient is negative or near zero
                        if dct_block[4, 3] > -1:
                            dct_block[4, 3] = -1
                    
                    # Apply inverse DCT
                    block = cv2.idct(dct_block)
                    
                    # Update image
                    y[i:i+8, j:j+8] = block
                    
                    bit_index += 1
                    bits_hidden_in_frame += 1
            
            # Merge channels back
            ycrcb = cv2.merge([y, cr, cb])
            
            # Convert back to BGR
            frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            
            # Save modified frame
            cv2.imwrite(frame_path, frame)
            
            if self.verbose:
                logger.debug(f"Hidden {bits_hidden_in_frame} bits in frame {frame_path}")
    
    def _decode_frame_dct(self, frame_paths: List[str]) -> List[int]:
        """
        Extract message from DCT coefficients method on frames.
        
        Args:
            frame_paths: Paths to frame images
            
        Returns:
            list: Extracted message bits
        """
        # Initialize extracted bits
        extracted_bits = []
        
        # Process each frame
        for frame_path in tqdm(frame_paths, disable=not self.verbose, desc="Decoding frames"):
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
            
            # Split channels
            y, _, _ = cv2.split(ycrcb)
            
            # Process 8x8 blocks in the Y channel
            height, width = y.shape
            
            for i in range(0, height - 8, 8):
                for j in range(0, width - 8, 8):
                    # Extract 8x8 block
                    block = y[i:i+8, j:j+8].astype(np.float32)
                    
                    # Apply DCT
                    dct_block = cv2.dct(block)
                    
                    # Extract bit from the mid-frequency coefficient (4,3)
                    if dct_block[4, 3] > 0:
                        extracted_bits.append(1)
                    else:
                        extracted_bits.append(0)
                    
                    # Check for termination sequence
                    if len(extracted_bits) >= 32 and all(bit == 0 for bit in extracted_bits[-32:]):
                        return extracted_bits[:-32]  # Remove termination sequence
        
        return extracted_bits
    
    def _encode_temporal(self, frame_paths: List[str], message_bits: List[int]) -> None:
        """
        Hide message using temporal patterns across frames.
        
        Args:
            frame_paths: Paths to frame images
            message_bits: Message bits to hide
        """
        # Ensure we have enough frames
        if len(frame_paths) < len(message_bits) + 2:
            raise ValueError(f"Not enough frames ({len(frame_paths)}) to encode message ({len(message_bits)} bits)")
        
        # We'll use subtle brightness changes between consecutive frames
        # 0 bit: make second frame slightly darker
        # 1 bit: make second frame slightly brighter
        
        # Process frames in pairs
        bit_index = 0
        
        for i in range(len(frame_paths) - 1):
            if bit_index >= len(message_bits):
                break
            
            # Read consecutive frames
            frame1 = cv2.imread(frame_paths[i])
            frame2 = cv2.imread(frame_paths[i + 1])
            
            if frame1 is None or frame2 is None:
                continue
            
            # Convert to HSV for easier brightness manipulation
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            
            # Modify brightness based on bit
            if message_bits[bit_index] == 0:
                # Make second frame slightly darker
                hsv2[:, :, 2] = np.clip(hsv2[:, :, 2] - 2, 0, 255)
            else:
                # Make second frame slightly brighter
                hsv2[:, :, 2] = np.clip(hsv2[:, :, 2] + 2, 0, 255)
            
            # Convert back to BGR
            frame2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
            
            # Save modified frame
            cv2.imwrite(frame_paths[i + 1], frame2)
            
            bit_index += 1
    
    def _decode_temporal(self, frame_paths: List[str]) -> List[int]:
        """
        Extract message from temporal patterns across frames.
        
        Args:
            frame_paths: Paths to frame images
            
        Returns:
            list: Extracted message bits
        """
        # Initialize extracted bits
        extracted_bits = []
        
        # Process frames in pairs
        for i in range(len(frame_paths) - 1):
            # Read consecutive frames
            frame1 = cv2.imread(frame_paths[i])
            frame2 = cv2.imread(frame_paths[i + 1])
            
            if frame1 is None or frame2 is None:
                continue
            
            # Convert to HSV
            hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
            
            # Compare brightness
            brightness1 = np.mean(hsv1[:, :, 2])
            brightness2 = np.mean(hsv2[:, :, 2])
            
            # Determine bit based on brightness difference
            if brightness2 < brightness1:
                extracted_bits.append(0)
            else:
                extracted_bits.append(1)
            
            # Check for termination sequence
            if len(extracted_bits) >= 32 and all(bit == 0 for bit in extracted_bits[-32:]):
                return extracted_bits[:-32]  # Remove termination sequence
        
        return extracted_bits
    
    def _encode_neural(self, frame_paths: List[str], message_bits: List[int]) -> None:
        """
        Hide message using neural image steganography on key frames.
        
        Args:
            frame_paths: Paths to frame images
            message_bits: Message bits to hide
        """
        # Check if we have an image model
        if self.image_model is None:
            self.image_model = ImageStegoNet(
                data_depth=self.data_depth,
                architecture='dense',
                cuda=self.cuda,
                verbose=self.verbose
            )
        
        # Calculate total bits to hide
        total_bits = len(message_bits)
        
        # Estimate bits per frame
        bits_per_frame = 10000  # Conservative estimate
        
        # Group frames for encoding
        frame_groups = []
        start_idx = 0
        
        while start_idx < len(message_bits):
            end_idx = min(start_idx + bits_per_frame, len(message_bits))
            frame_groups.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Ensure we have enough frames
        if len(frame_groups) > len(frame_paths):
            raise ValueError(f"Not enough frames ({len(frame_paths)}) to encode message")
        
        # Hide message bits in frames
        for i, (start_idx, end_idx) in enumerate(frame_groups):
            if i >= len(frame_paths):
                break
            
            frame_path = frame_paths[i]
            frame_bits = message_bits[start_idx:end_idx]
            
            # Use image model to hide bits in frame
            try:
                # Create temporary file paths
                temp_cover = frame_path
                temp_output = f"{frame_path}.stego.png"
                
                # Prepare message in binary format
                frame_message = ''.join(str(bit) for bit in frame_bits)
                
                # Use image model to encode
                self.image_model.encode(temp_cover, temp_output, frame_message)
                
                # Replace original frame with steganographic frame
                os.replace(temp_output, frame_path)
                
                if self.verbose:
                    logger.debug(f"Hidden {len(frame_bits)} bits in frame {i}")
            
            except Exception as e:
                logger.error(f"Error encoding frame {i}: {e}")
                continue
    
    def _decode_neural(self, frame_paths: List[str]) -> List[int]:
        """
        Extract message from neural image steganography on key frames.
        
        Args:
            frame_paths: Paths to frame images
            
        Returns:
            list: Extracted message bits
        """
        # Check if we have an image model
        if self.image_model is None:
            self.image_model = ImageStegoNet(
                data_depth=self.data_depth,
                architecture='dense',
                cuda=self.cuda,
                verbose=self.verbose
            )
        
        # Initialize extracted bits
        all_bits = []
        
        # Process each frame
        for i, frame_path in enumerate(tqdm(frame_paths, disable=not self.verbose, desc="Decoding frames")):
            try:
                # Use image model to extract message
                frame_message = self.image_model.decode(frame_path)
                
                # Convert message to bits
                frame_bits = [int(bit) for bit in frame_message if bit in '01']
                
                # Add to collected bits
                all_bits.extend(frame_bits)
                
                # Check for termination sequence
                if len(all_bits) >= 32 and all(bit == 0 for bit in all_bits[-32:]):
                    return all_bits[:-32]  # Remove termination sequence
            
            except Exception as e:
                logger.warning(f"Error decoding frame {i}: {e}")
                continue
        
        return all_bits
    
    def fit(self, *args, **kwargs):
        """
        Train the model.
        
        Video steganography typically doesn't require training, except for neural method.
        """
        if self.method == 'neural' and self.image_model:
            # Train the underlying image model
            return self.image_model.fit(*args, **kwargs)
        else:
            logger.info(f"No training needed for {self.method} method")
            return None
    
    @classmethod
    def load(cls, path=None, method='frame_lsb', cuda=True, verbose=False):
        """
        Load a model from disk or create a new one.
        
        Args:
            path: Path to saved model (optional)
            method: Steganography method (default: 'frame_lsb')
            cuda: Whether to use GPU if available
            verbose: Whether to print verbose output
            
        Returns:
            VideoStegoNet: Loaded or new model
        """
        if path is not None:
            try:
                model = torch.load(path, map_location='cpu')
                model.verbose = verbose
                
                if verbose:
                    logger.info(f"Model loaded from {path}")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                logger.info("Creating new model...")
        
        # Create new model with specified parameters
        return cls(method=method, cuda=cuda, verbose=verbose)