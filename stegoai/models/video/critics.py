# -*- coding: utf-8 -*-
"""
Video steganography critics for Stego-AI.

This module implements critic networks for detecting hidden messages
in videos using different steganographic techniques.
"""

import os
import logging
import tempfile
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from stegoai.models.base import BaseCritic
from stegoai.models.image.critics import ImageStegoCritic  # For neural method

# Set up logging
logger = logging.getLogger(__name__)


class VideoFrameLSBCritic(BaseCritic):
    """
    Critic for detecting LSB steganography in video frames.
    
    This critic analyzes frames for statistical anomalies in the
    least significant bits that indicate hidden data.
    """
    
    def __init__(self, cuda: bool = True):
        """
        Initialize the Frame LSB critic.
        
        Args:
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.cuda = cuda
    
    def forward(self, video_path: str, sample_rate: int = 10, max_frames: int = 30) -> Dict[str, Union[float, str]]:
        """
        Analyze a video for LSB steganography.
        
        Args:
            video_path: Path to the video file
            sample_rate: Only analyze every Nth frame
            max_frames: Maximum number of frames to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Extract sample frames
        frames = self._extract_sample_frames(video_path, sample_rate, max_frames)
        
        if not frames:
            return {
                'probability': 0.0,
                'assessment': "Could not analyze video",
                'confidence': "Low",
                'details': {
                    'frame_count': 0,
                    'lsb_anomaly_score': 0.0,
                    'lsb_correlation_score': 0.0,
                    'rs_detection_score': 0.0,
                }
            }
        
        # Calculate LSB anomaly score
        lsb_anomaly_score = self._calculate_lsb_anomaly_score(frames)
        
        # Calculate LSB correlation score (between color channels)
        lsb_correlation_score = self._calculate_lsb_correlation_score(frames)
        
        # Calculate RS detection score
        rs_detection_score = self._calculate_rs_detection_score(frames)
        
        # Combine scores
        combined_score = 0.4 * lsb_anomaly_score + 0.3 * lsb_correlation_score + 0.3 * rs_detection_score
        
        # Calculate confidence
        confidence = "Low"
        if len(frames) >= 10:
            confidence = "Medium"
        if len(frames) >= 20:
            confidence = "High"
        
        # Determine assessment
        if combined_score > 0.7:
            assessment = "Likely contains hidden data"
            probability = combined_score
        elif combined_score > 0.4:
            assessment = "Possibly contains hidden data"
            probability = combined_score
        else:
            assessment = "Likely clean"
            probability = combined_score
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'frame_count': len(frames),
                'lsb_anomaly_score': lsb_anomaly_score,
                'lsb_correlation_score': lsb_correlation_score,
                'rs_detection_score': rs_detection_score,
            }
        }
    
    def _extract_sample_frames(self, video_path: str, sample_rate: int, max_frames: int) -> List[np.ndarray]:
        """
        Extract sample frames from video for analysis.
        
        Args:
            video_path: Path to the video file
            sample_rate: Only extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            list: Extracted frames as numpy arrays
        """
        frames = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate sample indices
            if total_frames <= max_frames * sample_rate:
                # Extract every Nth frame
                indices = list(range(0, total_frames, sample_rate))
            else:
                # Extract frames evenly distributed
                step = total_frames / max_frames
                indices = [int(i * step) for i in range(max_frames)]
            
            # Limit to max_frames
            indices = indices[:max_frames]
            
            # Extract frames
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if success:
                    frames.append(frame)
            
            # Release video
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def _calculate_lsb_anomaly_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate anomaly score based on LSB statistics.
        
        Args:
            frames: List of video frames
            
        Returns:
            float: Anomaly score (0-1, higher means more likely to contain hidden data)
        """
        if not frames:
            return 0.0
        
        scores = []
        
        for frame in frames:
            # Extract least significant bits
            lsb_planes = []
            for channel in range(3):  # RGB
                channel_data = frame[:, :, channel]
                lsb_plane = channel_data & 1
                lsb_planes.append(lsb_plane)
            
            # Calculate histogram
            histograms = []
            for lsb_plane in lsb_planes:
                hist = np.bincount(lsb_plane.flatten(), minlength=2) / lsb_plane.size
                histograms.append(hist)
            
            # Check for abnormal distribution (ideally 50/50 for natural images)
            anomaly = 0.0
            for hist in histograms:
                # Calculate how far from 0.5 the distribution is
                anomaly += abs(hist[0] - 0.5) * 2  # Normalize to 0-1
            
            anomaly /= len(histograms)
            
            # For hidden data, anomaly should be close to 0 (random distribution)
            # For clean images, anomaly is often higher
            score = 1.0 - anomaly
            
            scores.append(score)
        
        # Return average score
        return sum(scores) / len(scores)
    
    def _calculate_lsb_correlation_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate correlation score between LSB planes.
        
        Args:
            frames: List of video frames
            
        Returns:
            float: Correlation score (0-1, higher means more likely to contain hidden data)
        """
        if not frames:
            return 0.0
        
        scores = []
        
        for frame in frames:
            # Extract least significant bits
            lsb_planes = []
            for channel in range(3):  # RGB
                channel_data = frame[:, :, channel]
                lsb_plane = channel_data & 1
                lsb_planes.append(lsb_plane)
            
            # Calculate correlations between LSB planes
            correlations = []
            for i in range(len(lsb_planes)):
                for j in range(i+1, len(lsb_planes)):
                    # Calculate correlation coefficient
                    plane1 = lsb_planes[i].astype(float)
                    plane2 = lsb_planes[j].astype(float)
                    
                    # Flatten arrays
                    plane1 = plane1.flatten()
                    plane2 = plane2.flatten()
                    
                    # Calculate correlation
                    try:
                        correlation = abs(np.corrcoef(plane1, plane2)[0, 1])
                        if np.isnan(correlation):
                            correlation = 0.0
                    except:
                        correlation = 0.0
                    
                    correlations.append(correlation)
            
            if correlations:
                # Average correlation
                avg_correlation = sum(correlations) / len(correlations)
                
                # For hidden data, correlation should be close to 0
                # For clean images, correlation is often higher
                score = 1.0 - avg_correlation
                
                scores.append(score)
        
        # Return average score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_rs_detection_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate RS (Regular/Singular) detection score.
        
        RS analysis is a statistical method to detect LSB steganography.
        
        Args:
            frames: List of video frames
            
        Returns:
            float: RS detection score (0-1, higher means more likely to contain hidden data)
        """
        if not frames:
            return 0.0
        
        scores = []
        
        # Define discriminant functions
        def f1(x, y):
            return 1 if x > y else -1 if x < y else 0
            
        def f_1(x, y):
            return -f1(x, y)
        
        for frame in frames:
            try:
                # Process each color channel
                channel_scores = []
                
                for channel in range(3):  # RGB
                    channel_data = frame[:, :, channel].copy()
                    height, width = channel_data.shape
                    
                    # Skip if too small
                    if height < 4 or width < 4:
                        continue
                    
                    # Create mask groups (work with 2x2 blocks)
                    masks = []
                    for i in range(0, height - 1, 2):
                        for j in range(0, width - 1, 2):
                            masks.append((i, j))
                    
                    # Randomly sample masks to speed up computation
                    if len(masks) > 500:
                        np.random.shuffle(masks)
                        masks = masks[:500]
                    
                    # Calculate R, S groups for original image
                    r_m, s_m = 0, 0
                    r_neg_m, s_neg_m = 0, 0
                    
                    for i, j in masks:
                        # Extract 2x2 block
                        block = channel_data[i:i+2, j:j+2].astype(int)
                        
                        # Calculate discriminant for original
                        discriminant = 0
                        for k in range(2):
                            for l in range(1, 2):
                                discriminant += f1(block[k, l-1], block[k, l])
                                discriminant += f1(block[l-1, k], block[l, k])
                        
                        # Flipped discriminant
                        discriminant_neg = 0
                        for k in range(2):
                            for l in range(1, 2):
                                discriminant_neg += f_1(block[k, l-1], block[k, l])
                                discriminant_neg += f_1(block[l-1, k], block[l, k])
                        
                        # Update counts
                        if discriminant > 0:
                            r_m += 1
                        elif discriminant < 0:
                            s_m += 1
                            
                        if discriminant_neg > 0:
                            r_neg_m += 1
                        elif discriminant_neg < 0:
                            s_neg_m += 1
                    
                    # Calculate LSB flipped version
                    flipped = channel_data.copy()
                    flipped = flipped ^ 1  # XOR with 1 flips the LSB
                    
                    # Calculate R, S groups for flipped image
                    r_m_flipped, s_m_flipped = 0, 0
                    r_neg_m_flipped, s_neg_m_flipped = 0, 0
                    
                    for i, j in masks:
                        # Extract 2x2 block from flipped image
                        block = flipped[i:i+2, j:j+2].astype(int)
                        
                        # Calculate discriminant for flipped
                        discriminant = 0
                        for k in range(2):
                            for l in range(1, 2):
                                discriminant += f1(block[k, l-1], block[k, l])
                                discriminant += f1(block[l-1, k], block[l, k])
                        
                        # Flipped discriminant
                        discriminant_neg = 0
                        for k in range(2):
                            for l in range(1, 2):
                                discriminant_neg += f_1(block[k, l-1], block[k, l])
                                discriminant_neg += f_1(block[l-1, k], block[l, k])
                        
                        # Update counts
                        if discriminant > 0:
                            r_m_flipped += 1
                        elif discriminant < 0:
                            s_m_flipped += 1
                            
                        if discriminant_neg > 0:
                            r_neg_m_flipped += 1
                        elif discriminant_neg < 0:
                            s_neg_m_flipped += 1
                    
                    # Calculate RS statistics
                    total_masks = len(masks)
                    if total_masks == 0:
                        continue
                        
                    rm = r_m / total_masks
                    sm = s_m / total_masks
                    rm_flipped = r_m_flipped / total_masks
                    sm_flipped = s_m_flipped / total_masks
                    
                    rm_neg = r_neg_m / total_masks
                    sm_neg = s_neg_m / total_masks
                    rm_neg_flipped = r_neg_m_flipped / total_masks
                    sm_neg_flipped = s_neg_m_flipped / total_masks
                    
                    # Calculate detection statistics
                    # For clean images: rm ≈ rm_flipped and sm ≈ sm_flipped
                    # For LSB steganography: rm ≠ rm_flipped and sm ≠ sm_flipped
                    
                    d1 = abs(rm - rm_flipped) + abs(sm - sm_flipped)
                    d2 = abs(rm_neg - rm_neg_flipped) + abs(sm_neg - sm_neg_flipped)
                    
                    # Normalize to 0-1
                    score = min(1.0, (d1 + d2) / 2)
                    channel_scores.append(score)
                
                # Average across channels
                if channel_scores:
                    scores.append(sum(channel_scores) / len(channel_scores))
                
            except Exception as e:
                logger.error(f"Error in RS detection: {e}")
        
        # Return average score
        return sum(scores) / len(scores) if scores else 0.0


class VideoFrameDCTCritic(BaseCritic):
    """
    Critic for detecting DCT-based steganography in video frames.
    
    This critic analyzes frames for anomalies in the frequency domain
    that might indicate hidden data in DCT coefficients.
    """
    
    def __init__(self, cuda: bool = True):
        """
        Initialize the Frame DCT critic.
        
        Args:
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.cuda = cuda
    
    def forward(self, video_path: str, sample_rate: int = 10, max_frames: int = 20) -> Dict[str, Union[float, str]]:
        """
        Analyze a video for DCT-based steganography.
        
        Args:
            video_path: Path to the video file
            sample_rate: Only analyze every Nth frame
            max_frames: Maximum number of frames to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Extract sample frames
        frames = self._extract_sample_frames(video_path, sample_rate, max_frames)
        
        if not frames:
            return {
                'probability': 0.0,
                'assessment': "Could not analyze video",
                'confidence': "Low",
                'details': {
                    'frame_count': 0,
                    'dct_histogram_score': 0.0,
                    'dct_calibration_score': 0.0,
                    'blockiness_score': 0.0,
                }
            }
        
        # Calculate DCT histogram score
        dct_histogram_score = self._calculate_dct_histogram_score(frames)
        
        # Calculate DCT calibration score
        dct_calibration_score = self._calculate_dct_calibration_score(frames)
        
        # Calculate blockiness score
        blockiness_score = self._calculate_blockiness_score(frames)
        
        # Combine scores
        combined_score = 0.4 * dct_histogram_score + 0.4 * dct_calibration_score + 0.2 * blockiness_score
        
        # Calculate confidence
        confidence = "Low"
        if len(frames) >= 10:
            confidence = "Medium"
        if len(frames) >= 15:
            confidence = "High"
        
        # Determine assessment
        if combined_score > 0.7:
            assessment = "Likely contains hidden data"
            probability = combined_score
        elif combined_score > 0.4:
            assessment = "Possibly contains hidden data"
            probability = combined_score
        else:
            assessment = "Likely clean"
            probability = combined_score
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'frame_count': len(frames),
                'dct_histogram_score': dct_histogram_score,
                'dct_calibration_score': dct_calibration_score,
                'blockiness_score': blockiness_score,
            }
        }
    
    def _extract_sample_frames(self, video_path: str, sample_rate: int, max_frames: int) -> List[np.ndarray]:
        """
        Extract sample frames from video for analysis.
        
        Args:
            video_path: Path to the video file
            sample_rate: Only extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            list: Extracted frames as numpy arrays
        """
        frames = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate sample indices
            if total_frames <= max_frames * sample_rate:
                # Extract every Nth frame
                indices = list(range(0, total_frames, sample_rate))
            else:
                # Extract frames evenly distributed
                step = total_frames / max_frames
                indices = [int(i * step) for i in range(max_frames)]
            
            # Limit to max_frames
            indices = indices[:max_frames]
            
            # Extract frames
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if success:
                    # Convert to grayscale for DCT analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(gray)
            
            # Release video
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def _calculate_dct_histogram_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate score based on DCT coefficient histogram.
        
        Args:
            frames: List of grayscale video frames
            
        Returns:
            float: DCT histogram score (0-1, higher means more likely to contain hidden data)
        """
        if not frames:
            return 0.0
        
        scores = []
        
        for frame in frames:
            try:
                height, width = frame.shape
                
                # Skip if too small
                if height < 8 or width < 8:
                    continue
                
                # Process 8x8 blocks
                histograms = []
                
                for i in range(0, height - 7, 8):
                    for j in range(0, width - 7, 8):
                        # Extract 8x8 block
                        block = frame[i:i+8, j:j+8].astype(np.float32)
                        
                        # Apply DCT
                        dct_block = cv2.dct(block)
                        
                        # Extract mid-frequency coefficients
                        # (where data is typically hidden)
                        mid_freq = []
                        for k in range(1, 8):
                            for l in range(8 - k):
                                mid_freq.append(dct_block[k, l])
                                mid_freq.append(dct_block[l, k])
                        
                        # Add to histograms
                        if mid_freq:
                            histograms.extend(mid_freq)
                
                if not histograms:
                    continue
                
                # Create histogram
                hist, _ = np.histogram(histograms, bins=50, range=(-50, 50), density=True)
                
                # Calculate smoothness of histogram
                # (hidden data tends to create irregularities)
                smoothness = 0.0
                for i in range(1, len(hist) - 1):
                    local_diff = abs(2 * hist[i] - hist[i-1] - hist[i+1])
                    smoothness += local_diff
                
                # Normalize smoothness
                smoothness /= len(hist)
                
                # Calculate score (higher smoothness indicates higher likelihood of hidden data)
                score = min(1.0, smoothness * 10)
                scores.append(score)
                
            except Exception as e:
                logger.error(f"Error in DCT histogram analysis: {e}")
        
        # Return average score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_dct_calibration_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate score based on DCT calibration analysis.
        
        This compares original frames with calibrated versions to detect anomalies.
        
        Args:
            frames: List of grayscale video frames
            
        Returns:
            float: DCT calibration score (0-1, higher means more likely to contain hidden data)
        """
        if not frames:
            return 0.0
        
        scores = []
        
        for frame in frames:
            try:
                height, width = frame.shape
                
                # Skip if too small
                if height < 10 or width < 10:
                    continue
                
                # Create calibrated image by cropping and recompressing
                # This is a common technique in steganalysis
                cropped = frame[1:height-1, 1:width-1]
                calibrated = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
                
                # Process 8x8 blocks
                diff_sum = 0.0
                block_count = 0
                
                for i in range(0, height - 7, 8):
                    for j in range(0, width - 7, 8):
                        # Extract 8x8 blocks
                        orig_block = frame[i:i+8, j:j+8].astype(np.float32)
                        calib_block = calibrated[i:i+8, j:j+8].astype(np.float32)
                        
                        # Apply DCT
                        orig_dct = cv2.dct(orig_block)
                        calib_dct = cv2.dct(calib_block)
                        
                        # Calculate difference in mid-frequency coefficients
                        diff = 0.0
                        count = 0
                        
                        for k in range(1, 8):
                            for l in range(8 - k):
                                diff += abs(orig_dct[k, l] - calib_dct[k, l])
                                diff += abs(orig_dct[l, k] - calib_dct[l, k])
                                count += 2
                        
                        if count > 0:
                            diff_sum += diff / count
                            block_count += 1
                
                if block_count == 0:
                    continue
                
                # Calculate average difference
                avg_diff = diff_sum / block_count
                
                # Convert to score (0-1)
                # Higher difference indicates higher likelihood of hidden data
                score = min(1.0, avg_diff / 10.0)
                scores.append(score)
                
            except Exception as e:
                logger.error(f"Error in DCT calibration analysis: {e}")
        
        # Return average score
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_blockiness_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate score based on blockiness analysis.
        
        This detects artifacts from DCT-based steganography that can cause
        increased blockiness at 8x8 block boundaries.
        
        Args:
            frames: List of grayscale video frames
            
        Returns:
            float: Blockiness score (0-1, higher means more likely to contain hidden data)
        """
        if not frames:
            return 0.0
        
        scores = []
        
        for frame in frames:
            try:
                height, width = frame.shape
                
                # Skip if too small
                if height < 16 or width < 16:
                    continue
                
                # Calculate blockiness at 8x8 boundaries
                h_block_diff = 0.0
                v_block_diff = 0.0
                h_nonblock_diff = 0.0
                v_nonblock_diff = 0.0
                
                # Horizontal blockiness
                h_block_count = 0
                h_nonblock_count = 0
                
                for i in range(height):
                    for j in range(1, width):
                        diff = abs(int(frame[i, j]) - int(frame[i, j-1]))
                        
                        # Check if we're at a block boundary
                        if j % 8 == 0:
                            h_block_diff += diff
                            h_block_count += 1
                        else:
                            h_nonblock_diff += diff
                            h_nonblock_count += 1
                
                # Vertical blockiness
                v_block_count = 0
                v_nonblock_count = 0
                
                for j in range(width):
                    for i in range(1, height):
                        diff = abs(int(frame[i, j]) - int(frame[i-1, j]))
                        
                        # Check if we're at a block boundary
                        if i % 8 == 0:
                            v_block_diff += diff
                            v_block_count += 1
                        else:
                            v_nonblock_diff += diff
                            v_nonblock_count += 1
                
                # Calculate average differences
                h_block_avg = h_block_diff / h_block_count if h_block_count > 0 else 0
                h_nonblock_avg = h_nonblock_diff / h_nonblock_count if h_nonblock_count > 0 else 0
                v_block_avg = v_block_diff / v_block_count if v_block_count > 0 else 0
                v_nonblock_avg = v_nonblock_diff / v_nonblock_count if v_nonblock_count > 0 else 0
                
                # Calculate blockiness ratios
                h_ratio = h_block_avg / h_nonblock_avg if h_nonblock_avg > 0 else 1
                v_ratio = v_block_avg / v_nonblock_avg if v_nonblock_avg > 0 else 1
                
                # Average ratio
                avg_ratio = (h_ratio + v_ratio) / 2
                
                # Convert to score (0-1)
                # Higher ratio indicates higher blockiness at block boundaries
                score = min(1.0, max(0.0, (avg_ratio - 1.0) / 0.5))
                scores.append(score)
                
            except Exception as e:
                logger.error(f"Error in blockiness analysis: {e}")
        
        # Return average score
        return sum(scores) / len(scores) if scores else 0.0


class VideoTemporalCritic(BaseCritic):
    """
    Critic for detecting temporal steganography in videos.
    
    This critic analyzes temporal patterns in video frames to
    detect hidden data encoded in frame differences or sequences.
    """
    
    def __init__(self, cuda: bool = True):
        """
        Initialize the Temporal critic.
        
        Args:
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.cuda = cuda
    
    def forward(self, video_path: str, max_frames: int = 100) -> Dict[str, Union[float, str]]:
        """
        Analyze a video for temporal steganography.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Extract consecutive frames
        frames = self._extract_consecutive_frames(video_path, max_frames)
        
        if not frames or len(frames) < 3:
            return {
                'probability': 0.0,
                'assessment': "Could not analyze video",
                'confidence': "Low",
                'details': {
                    'frame_count': len(frames) if frames else 0,
                    'brightness_pattern_score': 0.0,
                    'motion_pattern_score': 0.0,
                    'frequency_domain_score': 0.0,
                }
            }
        
        # Calculate brightness pattern score
        brightness_pattern_score = self._calculate_brightness_pattern_score(frames)
        
        # Calculate motion pattern score
        motion_pattern_score = self._calculate_motion_pattern_score(frames)
        
        # Calculate frequency domain score
        frequency_domain_score = self._calculate_frequency_domain_score(frames)
        
        # Combine scores
        combined_score = 0.4 * brightness_pattern_score + 0.4 * motion_pattern_score + 0.2 * frequency_domain_score
        
        # Calculate confidence
        confidence = "Low"
        if len(frames) >= 30:
            confidence = "Medium"
        if len(frames) >= 60:
            confidence = "High"
        
        # Determine assessment
        if combined_score > 0.7:
            assessment = "Likely contains hidden data"
            probability = combined_score
        elif combined_score > 0.4:
            assessment = "Possibly contains hidden data"
            probability = combined_score
        else:
            assessment = "Likely clean"
            probability = combined_score
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'frame_count': len(frames),
                'brightness_pattern_score': brightness_pattern_score,
                'motion_pattern_score': motion_pattern_score,
                'frequency_domain_score': frequency_domain_score,
            }
        }
    
    def _extract_consecutive_frames(self, video_path: str, max_frames: int) -> List[np.ndarray]:
        """
        Extract consecutive frames from video for analysis.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            list: Extracted frames as numpy arrays
        """
        frames = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_to_extract = min(total_frames, max_frames)
            
            # Extract consecutive frames from a random starting point
            if total_frames > max_frames:
                start_frame = np.random.randint(0, total_frames - max_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract frames
            for _ in range(frames_to_extract):
                success, frame = cap.read()
                if not success:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            
            # Release video
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
        
        return frames
    
    def _calculate_brightness_pattern_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate score based on brightness patterns between consecutive frames.
        
        Args:
            frames: List of grayscale video frames
            
        Returns:
            float: Brightness pattern score (0-1, higher means more likely to contain hidden data)
        """
        if not frames or len(frames) < 2:
            return 0.0
        
        try:
            # Calculate average brightness for each frame
            brightnesses = [frame.mean() for frame in frames]
            
            # Calculate frame-to-frame differences
            differences = [abs(brightnesses[i] - brightnesses[i-1]) for i in range(1, len(brightnesses))]
            
            # Look for alternating patterns (common in temporal steganography)
            alternating_count = 0
            
            for i in range(1, len(differences)):
                # Check if differences are alternating (one large, one small)
                if (differences[i] > differences[i-1] * 1.5) or (differences[i] * 1.5 < differences[i-1]):
                    alternating_count += 1
            
            alternating_ratio = alternating_count / (len(differences) - 1) if len(differences) > 1 else 0
            
            # Look for periodic patterns
            periodic_scores = []
            
            for period in range(2, min(10, len(differences) // 3)):
                periodicity = 0.0
                for i in range(period, len(differences)):
                    similarity = 1.0 - min(1.0, abs(differences[i] - differences[i-period]) / max(differences[i], differences[i-period], 0.1))
                    periodicity += similarity
                
                avg_periodicity = periodicity / (len(differences) - period) if len(differences) > period else 0
                periodic_scores.append(avg_periodicity)
            
            max_periodicity = max(periodic_scores) if periodic_scores else 0
            
            # Combine alternating and periodic scores
            score = 0.5 * alternating_ratio + 0.5 * max_periodicity
            
            return min(1.0, score * 1.5)  # Scale up slightly for better sensitivity
            
        except Exception as e:
            logger.error(f"Error calculating brightness pattern score: {e}")
            return 0.0
    
    def _calculate_motion_pattern_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate score based on motion patterns between consecutive frames.
        
        Args:
            frames: List of grayscale video frames
            
        Returns:
            float: Motion pattern score (0-1, higher means more likely to contain hidden data)
        """
        if not frames or len(frames) < 3:
            return 0.0
        
        try:
            # Calculate optical flow for consecutive frames
            flow_magnitudes = []
            
            for i in range(1, len(frames)):
                # Calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    frames[i-1], frames[i], None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # Calculate magnitude of flow
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_mag = mag.mean()
                flow_magnitudes.append(avg_mag)
            
            if not flow_magnitudes:
                return 0.0
            
            # Look for unusual patterns in flow
            # Calculate differences in flow magnitudes
            diff_magnitudes = [abs(flow_magnitudes[i] - flow_magnitudes[i-1]) for i in range(1, len(flow_magnitudes))]
            
            if not diff_magnitudes:
                return 0.0
            
            # Calculate variability in differences
            mean_diff = sum(diff_magnitudes) / len(diff_magnitudes)
            variance = sum((diff - mean_diff) ** 2 for diff in diff_magnitudes) / len(diff_magnitudes)
            std_dev = variance ** 0.5
            
            # Normalize standard deviation by mean to get coefficient of variation
            cv_value = std_dev / mean_diff if mean_diff > 0 else 0
            
            # Calculate score based on coefficient of variation
            # Higher variability indicates higher likelihood of hidden data
            cv_score = min(1.0, cv_value / 2.0)
            
            # Look for alternating patterns
            alternating_count = 0
            for i in range(1, len(diff_magnitudes)):
                if (diff_magnitudes[i] > diff_magnitudes[i-1] * 1.5) or (diff_magnitudes[i] * 1.5 < diff_magnitudes[i-1]):
                    alternating_count += 1
            
            alternating_ratio = alternating_count / (len(diff_magnitudes) - 1) if len(diff_magnitudes) > 1 else 0
            
            # Combine scores
            combined_score = 0.6 * cv_score + 0.4 * alternating_ratio
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error calculating motion pattern score: {e}")
            return 0.0
    
    def _calculate_frequency_domain_score(self, frames: List[np.ndarray]) -> float:
        """
        Calculate score based on frequency domain analysis of frame sequences.
        
        Args:
            frames: List of grayscale video frames
            
        Returns:
            float: Frequency domain score (0-1, higher means more likely to contain hidden data)
        """
        if not frames or len(frames) < 8:
            return 0.0
        
        try:
            # Extract time series from random pixels
            height, width = frames[0].shape
            num_points = min(100, height * width // 1000)  # Number of random points to sample
            
            # Generate random points
            points = []
            for _ in range(num_points):
                y = np.random.randint(0, height)
                x = np.random.randint(0, width)
                points.append((y, x))
            
            # Extract time series
            time_series = []
            for y, x in points:
                series = [int(frame[y, x]) for frame in frames]
                time_series.append(series)
            
            # Calculate frequency domain features
            fft_peaks = []
            
            for series in time_series:
                # Apply FFT
                fft = np.abs(np.fft.fft(series))
                
                # Ignore DC component
                fft = fft[1:len(fft)//2]
                
                # Find peaks
                if len(fft) > 2:
                    # Normalize
                    fft_norm = fft / np.mean(fft) if np.mean(fft) > 0 else fft
                    
                    # Find peaks that are significantly above average
                    peaks = [i for i in range(1, len(fft_norm)-1) if fft_norm[i] > fft_norm[i-1] and fft_norm[i] > fft_norm[i+1] and fft_norm[i] > 1.5]
                    
                    # Count significant peaks
                    peak_count = len(peaks)
                    fft_peaks.append(peak_count)
            
            if not fft_peaks:
                return 0.0
            
            # Average peak count
            avg_peaks = sum(fft_peaks) / len(fft_peaks)
            
            # Calculate score based on peak count
            # More peaks indicates higher likelihood of hidden data with periodic patterns
            score = min(1.0, avg_peaks / 3.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating frequency domain score: {e}")
            return 0.0


class VideoNeuralCritic(BaseCritic):
    """
    Critic for detecting neural steganography in videos.
    
    This critic uses image steganography detection techniques on
    individual frames to identify neural steganography in videos.
    """
    
    def __init__(self, cuda: bool = True):
        """
        Initialize the Neural critic.
        
        Args:
            cuda: Whether to use GPU acceleration if available
        """
        super().__init__()
        self.cuda = cuda
        
        # Initialize image critic
        try:
            self.image_critic = ImageStegoCritic(architecture='dense', cuda=cuda)
            self.image_critic_loaded = True
        except Exception as e:
            logger.error(f"Error loading image critic: {e}")
            self.image_critic_loaded = False
    
    def forward(self, video_path: str, sample_rate: int = 5, max_frames: int = 10) -> Dict[str, Union[float, str]]:
        """
        Analyze a video for neural steganography.
        
        Args:
            video_path: Path to the video file
            sample_rate: Only analyze every Nth frame
            max_frames: Maximum number of frames to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Check if image critic is available
        if not self.image_critic_loaded:
            return {
                'probability': 0.0,
                'assessment': "Could not analyze video (image critic not available)",
                'confidence': "Low",
                'details': {
                    'frame_count': 0,
                    'neural_stego_score': 0.0,
                    'consistency_score': 0.0,
                }
            }
        
        # Extract sample frames
        frame_paths = self._extract_sample_frames(video_path, sample_rate, max_frames)
        
        if not frame_paths:
            return {
                'probability': 0.0,
                'assessment': "Could not analyze video",
                'confidence': "Low",
                'details': {
                    'frame_count': 0,
                    'neural_stego_score': 0.0,
                    'consistency_score': 0.0,
                }
            }
        
        try:
            # Analyze each frame with the image critic
            frame_scores = []
            
            for frame_path in frame_paths:
                # Analyze frame
                result = self.image_critic.analyze_image(frame_path)
                
                # Extract probability
                probability = result.get('probability', 0.0)
                frame_scores.append(probability)
            
            # Calculate average score
            avg_score = sum(frame_scores) / len(frame_scores) if frame_scores else 0.0
            
            # Calculate consistency score
            # (neural steganography typically affects frames consistently)
            consistency = 0.0
            
            if len(frame_scores) >= 2:
                # Calculate standard deviation
                mean = avg_score
                variance = sum((score - mean) ** 2 for score in frame_scores) / len(frame_scores)
                std_dev = variance ** 0.5
                
                # Calculate coefficient of variation (lower is more consistent)
                cv = std_dev / mean if mean > 0 else 0
                
                # Convert to consistency score (0-1, higher is more consistent)
                consistency = 1.0 - min(1.0, cv)
            
            # Calculate combined score
            combined_score = 0.7 * avg_score + 0.3 * consistency
            
            # Calculate confidence
            confidence = "Low"
            if len(frame_paths) >= 5:
                confidence = "Medium"
            if len(frame_paths) >= 8:
                confidence = "High"
            
            # Determine assessment
            if combined_score > 0.7:
                assessment = "Likely contains hidden data"
                probability = combined_score
            elif combined_score > 0.4:
                assessment = "Possibly contains hidden data"
                probability = combined_score
            else:
                assessment = "Likely clean"
                probability = combined_score
            
            # Clean up temporary files
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            
            return {
                'probability': probability,
                'assessment': assessment,
                'confidence': confidence,
                'details': {
                    'frame_count': len(frame_paths),
                    'neural_stego_score': avg_score,
                    'consistency_score': consistency,
                    'frame_scores': frame_scores,
                }
            }
            
        except Exception as e:
            logger.error(f"Error in neural stego analysis: {e}")
            
            # Clean up temporary files
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
            
            return {
                'probability': 0.0,
                'assessment': f"Error analyzing video: {e}",
                'confidence': "Low",
                'details': {
                    'frame_count': len(frame_paths),
                    'neural_stego_score': 0.0,
                    'consistency_score': 0.0,
                }
            }
    
    def _extract_sample_frames(self, video_path: str, sample_rate: int, max_frames: int) -> List[str]:
        """
        Extract sample frames from video and save to temporary files.
        
        Args:
            video_path: Path to the video file
            sample_rate: Only extract every Nth frame
            max_frames: Maximum number of frames to extract
            
        Returns:
            list: Paths to extracted frame image files
        """
        frame_paths = []
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return []
            
            # Get total frame count
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate sample indices
            if total_frames <= max_frames * sample_rate:
                # Extract every Nth frame
                indices = list(range(0, total_frames, sample_rate))
            else:
                # Extract frames evenly distributed
                step = total_frames / max_frames
                indices = [int(i * step) for i in range(max_frames)]
            
            # Limit to max_frames
            indices = indices[:max_frames]
            
            # Extract frames
            for i, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if success:
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f"frame_{i:06d}.png")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
            
            # Release video
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            
            # Clean up on error
            for path in frame_paths:
                try:
                    os.remove(path)
                except:
                    pass
            
            frame_paths = []
        
        return frame_paths


def get_video_critic(method: str = 'frame_lsb', **kwargs) -> BaseCritic:
    """
    Factory function to get the appropriate video critic.
    
    Args:
        method: Steganography method ('frame_lsb', 'frame_dct', 'temporal', 'neural')
        **kwargs: Additional arguments to pass to the critic
        
    Returns:
        BaseCritic: Appropriate critic for the method
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'frame_lsb':
        return VideoFrameLSBCritic(**kwargs)
    elif method == 'frame_dct':
        return VideoFrameDCTCritic(**kwargs)
    elif method == 'temporal':
        return VideoTemporalCritic(**kwargs)
    elif method == 'neural':
        return VideoNeuralCritic(**kwargs)
    else:
        raise ValueError(f"Unsupported video steganography method: {method}")