# -*- coding: utf-8 -*-
"""
Video processing utilities for Stego-AI.

This module provides functions for loading, saving, and processing
video files used in steganographic operations.
"""

import os
import tempfile
import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


def check_ffmpeg() -> bool:
    """
    Check if FFmpeg is available on the system.
    
    Returns:
        bool: True if FFmpeg is available, False otherwise
    """
    try:
        subprocess.check_output(['ffmpeg', '-version'], stderr=subprocess.STDOUT)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("FFmpeg not found. Some functionality may be limited.")
        return False


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video information including width, height, fps, duration, etc.
    """
    try:
        # Check if FFmpeg is available
        if check_ffmpeg():
            # Use FFmpeg for more accurate information
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration,bit_rate,codec_name',
                '-show_entries', 'format=duration,size,bit_rate',
                '-of', 'json',
                video_path
            ]
            
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            import json
            info = json.loads(output)
            
            # Extract stream information
            stream_info = info.get('streams', [{}])[0]
            format_info = info.get('format', {})
            
            # Parse frame rate (could be a fraction)
            fps_str = stream_info.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, denom = map(int, fps_str.split('/'))
                fps = num / denom if denom else 0
            else:
                fps = float(fps_str)
            
            # Get duration
            duration = float(stream_info.get('duration', format_info.get('duration', 0)))
            
            # Calculate total frames
            total_frames = int(duration * fps) if duration and fps else 0
            
            # Create result dictionary
            result = {
                'width': int(stream_info.get('width', 0)),
                'height': int(stream_info.get('height', 0)),
                'fps': fps,
                'duration': duration,
                'total_frames': total_frames,
                'codec': stream_info.get('codec_name', ''),
                'bitrate': int(stream_info.get('bit_rate', format_info.get('bit_rate', 0))),
                'size_bytes': int(format_info.get('size', 0)),
            }
            
            return result
        else:
            # Fall back to OpenCV
            return get_video_info_cv2(video_path)
    
    except Exception as e:
        logger.error(f"Error getting video info with FFmpeg: {e}")
        # Fall back to OpenCV
        return get_video_info_cv2(video_path)


def get_video_info_cv2(video_path: str) -> Dict[str, Any]:
    """
    Get information about a video file using OpenCV.
    
    Args:
        video_path: Path to video file
        
    Returns:
        dict: Video information
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        duration = total_frames / fps if fps > 0 else 0
        
        # Get codec (four character code)
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # File size
        size_bytes = os.path.getsize(video_path)
        
        # Calculate bitrate (bits per second)
        bitrate = int((size_bytes * 8) / duration) if duration > 0 else 0
        
        cap.release()
        
        return {
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames,
            'codec': codec,
            'bitrate': bitrate,
            'size_bytes': size_bytes,
        }
    
    except Exception as e:
        logger.error(f"Error getting video info with OpenCV: {e}")
        # Return empty info on error
        return {
            'width': 0,
            'height': 0,
            'fps': 0,
            'duration': 0,
            'total_frames': 0,
            'codec': '',
            'bitrate': 0,
            'size_bytes': 0,
        }


def extract_frames(video_path: str, output_dir: str, 
                   frame_indices: Optional[List[int]] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   fps: Optional[float] = None) -> List[str]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_indices: Specific frame indices to extract (None for all)
        start_time: Start time in seconds (None for beginning)
        end_time: End time in seconds (None for end)
        fps: Extract at specific fps (None for original fps)
        
    Returns:
        list: Paths to extracted frame images
    """
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if FFmpeg is available
        if check_ffmpeg() and frame_indices is None:
            # Use FFmpeg for more efficient frame extraction
            return extract_frames_ffmpeg(video_path, output_dir, start_time, end_time, fps)
        else:
            # Use OpenCV for more precise frame extraction
            return extract_frames_cv2(video_path, output_dir, frame_indices, start_time, end_time, fps)
    
    except Exception as e:
        logger.error(f"Error extracting frames: {e}")
        raise


def extract_frames_ffmpeg(video_path: str, output_dir: str,
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None,
                         fps: Optional[float] = None) -> List[str]:
    """
    Extract frames using FFmpeg.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Extract at specific fps
        
    Returns:
        list: Paths to extracted frame images
    """
    cmd = ['ffmpeg', '-y', '-i', video_path]
    
    # Add start time if specified
    if start_time is not None:
        cmd.extend(['-ss', str(start_time)])
    
    # Add end time if specified
    if end_time is not None:
        cmd.extend(['-to', str(end_time)])
    
    # Add fps if specified
    if fps is not None:
        cmd.extend(['-r', str(fps)])
    
    # Output pattern
    output_pattern = os.path.join(output_dir, 'frame_%06d.png')
    cmd.append(output_pattern)
    
    # Run FFmpeg command
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)
    
    # Get paths of extracted frames
    frame_paths = sorted([
        os.path.join(output_dir, f) for f in os.listdir(output_dir)
        if f.startswith('frame_') and f.endswith('.png')
    ])
    
    return frame_paths


def extract_frames_cv2(video_path: str, output_dir: str,
                      frame_indices: Optional[List[int]] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      fps: Optional[float] = None) -> List[str]:
    """
    Extract frames using OpenCV.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save extracted frames
        frame_indices: Specific frame indices to extract
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Extract at specific fps
        
    Returns:
        list: Paths to extracted frame images
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate start and end frames
    start_frame = 0
    if start_time is not None:
        start_frame = int(start_time * original_fps)
    
    end_frame = total_frames
    if end_time is not None:
        end_frame = min(total_frames, int(end_time * original_fps))
    
    # Calculate frame step based on fps
    frame_step = 1
    if fps is not None and fps < original_fps:
        frame_step = max(1, int(original_fps / fps))
    
    # Determine frames to extract
    if frame_indices is None:
        frame_indices = list(range(start_frame, end_frame, frame_step))
    else:
        # Filter indices to be within range
        frame_indices = [i for i in frame_indices if start_frame <= i < end_frame]
    
    # Extract frames
    frame_paths = []
    
    # Set initial position
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    current_frame = start_frame
    
    with tqdm(total=len(frame_indices), desc="Extracting frames") as pbar:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame in frame_indices:
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{current_frame:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                pbar.update(1)
            
            current_frame += 1
    
    cap.release()
    
    return frame_paths


def create_video(frame_paths: List[str], output_path: str, fps: float = 30, 
                 width: Optional[int] = None, height: Optional[int] = None,
                 quality: int = 23, codec: str = 'libx264') -> str:
    """
    Create video from frames.
    
    Args:
        frame_paths: Paths to frame images
        output_path: Path to output video
        fps: Frames per second
        width: Video width (None for auto)
        height: Video height (None for auto)
        quality: Video quality (CRF value, lower is better)
        codec: Video codec
        
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
        if check_ffmpeg():
            # Use FFmpeg to create video
            return create_video_ffmpeg(frame_paths, output_path, fps, width, height, quality, codec)
        else:
            # Use OpenCV to create video
            return create_video_cv2(frame_paths, output_path, fps, width, height)
    
    except Exception as e:
        logger.error(f"Error creating video: {e}")
        raise


def create_video_ffmpeg(frame_paths: List[str], output_path: str, fps: float = 30,
                       width: int = 1920, height: int = 1080,
                       quality: int = 23, codec: str = 'libx264') -> str:
    """
    Create video from frames using FFmpeg.
    
    Args:
        frame_paths: Paths to frame images
        output_path: Path to output video
        fps: Frames per second
        width: Video width
        height: Video height
        quality: Video quality (CRF value, lower is better)
        codec: Video codec
        
    Returns:
        str: Path to created video
    """
    # Create directory structure
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Get frame pattern from first frame
    first_frame = os.path.basename(frame_paths[0])
    frame_dir = os.path.dirname(frame_paths[0])
    
    # Extract frame number pattern
    import re
    match = re.search(r'(\d+)', first_frame)
    if not match:
        raise ValueError(f"Invalid frame filename pattern: {first_frame}")
    
    num_digits = len(match.group(1))
    prefix = first_frame[:match.start()]
    suffix = first_frame[match.end():]
    
    # Create frame pattern
    pattern = f"{prefix}%0{num_digits}d{suffix}"
    
    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),  # Input frame rate
        '-i', os.path.join(frame_dir, pattern),  # Input pattern
        '-c:v', codec,  # Codec
        '-crf', str(quality),  # Quality
        '-preset', 'medium',  # Encoding speed/quality tradeoff
        '-pix_fmt', 'yuv420p',  # Pixel format
        '-vf', f'scale={width}:{height}',  # Scale to desired dimensions
        output_path
    ]
    
    # Run FFmpeg command
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)
    
    return output_path


def create_video_cv2(frame_paths: List[str], output_path: str, fps: float = 30,
                    width: int = 1920, height: int = 1080) -> str:
    """
    Create video from frames using OpenCV.
    
    Args:
        frame_paths: Paths to frame images
        output_path: Path to output video
        fps: Frames per second
        width: Video width
        height: Video height
        
    Returns:
        str: Path to created video
    """
    # Create directory structure
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Determine codec based on output file extension
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        # Default to MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    with tqdm(total=len(frame_paths), desc="Creating video") as pbar:
        for frame_path in frame_paths:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                continue
            
            # Resize frame if needed
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            
            # Write frame to video
            out.write(frame)
            pbar.update(1)
    
    # Release video writer
    out.release()
    
    return output_path


def split_video(video_path: str, output_dir: str, segment_duration: float = 10.0) -> List[str]:
    """
    Split video into segments.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save segments
        segment_duration: Duration of each segment in seconds
        
    Returns:
        list: Paths to video segments
    """
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video info
        video_info = get_video_info(video_path)
        duration = video_info['duration']
        
        # Calculate number of segments
        num_segments = max(1, int(duration / segment_duration))
        
        # Check if FFmpeg is available
        if check_ffmpeg():
            # Use FFmpeg for splitting
            segment_paths = []
            
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, duration)
                
                output_path = os.path.join(output_dir, f"segment_{i:03d}.mp4")
                
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-to', str(end_time),
                    '-c', 'copy',  # Copy codec without re-encoding
                    output_path
                ]
                
                subprocess.check_call(cmd, stderr=subprocess.STDOUT)
                segment_paths.append(output_path)
            
            return segment_paths
        else:
            # Use OpenCV for splitting
            segment_paths = []
            
            # Get video properties
            fps = video_info['fps']
            width = video_info['width']
            height = video_info['height']
            
            # Calculate frames per segment
            frames_per_segment = int(segment_duration * fps)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            segment_idx = 0
            frame_count = 0
            out = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Start new segment if needed
                if frame_count % frames_per_segment == 0:
                    # Close previous segment
                    if out is not None:
                        out.release()
                    
                    # Create new segment
                    output_path = os.path.join(output_dir, f"segment_{segment_idx:03d}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    segment_paths.append(output_path)
                    segment_idx += 1
                
                # Write frame to current segment
                out.write(frame)
                frame_count += 1
            
            # Close last segment
            if out is not None:
                out.release()
            
            cap.release()
            
            return segment_paths
    
    except Exception as e:
        logger.error(f"Error splitting video: {e}")
        raise


def merge_videos(video_paths: List[str], output_path: str, transition: str = 'none') -> str:
    """
    Merge multiple videos into one.
    
    Args:
        video_paths: Paths to input videos
        output_path: Path to output video
        transition: Transition type ('none', 'fade', 'dissolve')
        
    Returns:
        str: Path to merged video
    """
    try:
        # Check if FFmpeg is available
        if not check_ffmpeg():
            raise ValueError("FFmpeg is required for merging videos")
        
        # Create temp file for concat list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            concat_file = f.name
            
            # Write file paths to concat list
            for video_path in video_paths:
                f.write(f"file '{os.path.abspath(video_path)}'\n")
        
        # Create directory for output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Build FFmpeg command based on transition type
        if transition == 'none':
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',  # Copy codec without re-encoding
                output_path
            ]
        else:
            # For transitions, we need to re-encode
            # This is a simplified approach, actual implementation would be more complex
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c:v', 'libx264',
                '-crf', '23',
                '-preset', 'medium',
                output_path
            ]
        
        # Run FFmpeg command
        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        
        # Clean up concat file
        os.unlink(concat_file)
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error merging videos: {e}")
        raise


def apply_filter(video_path: str, output_path: str, filter_type: str,
                parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply filter to video.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        filter_type: Filter type ('blur', 'sharpen', 'noise', etc.)
        parameters: Filter parameters
        
    Returns:
        str: Path to filtered video
    """
    try:
        # Initialize parameters dictionary
        if parameters is None:
            parameters = {}
        
        # Check if FFmpeg is available
        if check_ffmpeg():
            # Use FFmpeg for filtering
            cmd = ['ffmpeg', '-y', '-i', video_path]
            
            # Apply appropriate filter
            if filter_type == 'blur':
                # Get blur amount (default: 5)
                amount = parameters.get('amount', 5)
                cmd.extend(['-vf', f'boxblur={amount}:1'])
            
            elif filter_type == 'sharpen':
                # Get sharpen amount (default: 1.5)
                amount = parameters.get('amount', 1.5)
                cmd.extend(['-vf', f'unsharp=5:5:{amount}:5:5:{amount}'])
            
            elif filter_type == 'noise':
                # Get noise amount (default: 10)
                amount = parameters.get('amount', 10)
                cmd.extend(['-vf', f'noise=alls={amount}:allf=t'])
            
            elif filter_type == 'brightness':
                # Get brightness adjustment (default: 0.1)
                amount = parameters.get('amount', 0.1)
                cmd.extend(['-vf', f'eq=brightness={amount}'])
            
            elif filter_type == 'contrast':
                # Get contrast adjustment (default: 1.1)
                amount = parameters.get('amount', 1.1)
                cmd.extend(['-vf', f'eq=contrast={amount}'])
            
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
            
            # Add output path
            cmd.append(output_path)
            
            # Run FFmpeg command
            subprocess.check_call(cmd, stderr=subprocess.STDOUT)
            
            return output_path
        else:
            # Use OpenCV for filtering
            return apply_filter_cv2(video_path, output_path, filter_type, parameters)
    
    except Exception as e:
        logger.error(f"Error applying filter: {e}")
        raise


def apply_filter_cv2(video_path: str, output_path: str, filter_type: str,
                    parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply filter to video using OpenCV.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        filter_type: Filter type ('blur', 'sharpen', 'noise', etc.)
        parameters: Filter parameters
        
    Returns:
        str: Path to filtered video
    """
    # Initialize parameters dictionary
    if parameters is None:
        parameters = {}
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    with tqdm(desc="Applying filter") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Apply filter based on type
            if filter_type == 'blur':
                # Get blur amount (default: 5)
                amount = parameters.get('amount', 5)
                processed = cv2.GaussianBlur(frame, (amount * 2 + 1, amount * 2 + 1), 0)
            
            elif filter_type == 'sharpen':
                # Get sharpen amount (default: 1.5)
                amount = parameters.get('amount', 1.5)
                kernel = np.array([[-1, -1, -1], [-1, 9 + (amount - 1) * 4, -1], [-1, -1, -1]])
                processed = cv2.filter2D(frame, -1, kernel)
            
            elif filter_type == 'noise':
                # Get noise amount (default: 10)
                amount = parameters.get('amount', 10)
                noise = np.random.randn(*frame.shape) * amount
                processed = cv2.add(frame, noise.astype(np.uint8))
            
            elif filter_type == 'brightness':
                # Get brightness adjustment (default: 0.1)
                amount = parameters.get('amount', 0.1)
                processed = cv2.convertScaleAbs(frame, alpha=1, beta=amount * 255)
            
            elif filter_type == 'contrast':
                # Get contrast adjustment (default: 1.1)
                amount = parameters.get('amount', 1.1)
                processed = cv2.convertScaleAbs(frame, alpha=amount, beta=0)
            
            else:
                processed = frame
            
            # Write processed frame
            out.write(processed)
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path


def video_to_frames(video_path: str) -> np.ndarray:
    """
    Load video and convert to frames array.
    
    Args:
        video_path: Path to video file
        
    Returns:
        numpy.ndarray: Frames array (frames, height, width, channels)
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Read frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames read from video")
        
        # Convert to numpy array
        return np.array(frames)
    
    except Exception as e:
        logger.error(f"Error converting video to frames: {e}")
        raise


def frames_to_video(frames: np.ndarray, output_path: str, fps: float = 30) -> str:
    """
    Convert frames array to video.
    
    Args:
        frames: Frames array (frames, height, width, channels)
        output_path: Path to output video
        fps: Frames per second
        
    Returns:
        str: Path to created video
    """
    try:
        # Create directory for output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get dimensions
        if len(frames.shape) != 4:
            raise ValueError(f"Invalid frames shape: {frames.shape}")
        
        num_frames, height, width, channels = frames.shape
        
        # Check channels
        if channels != 3:
            raise ValueError(f"Expected 3 channels, got {channels}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for i in range(num_frames):
            # Convert RGB to BGR
            frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error converting frames to video: {e}")
        raise


def compress_video(input_path: str, output_path: str, quality: int = 23,
                  width: Optional[int] = None, height: Optional[int] = None) -> str:
    """
    Compress video to reduce file size.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        quality: Compression quality (CRF value, lower is better)
        width: Output width (None for original)
        height: Output height (None for original)
        
    Returns:
        str: Path to compressed video
    """
    try:
        # Check if FFmpeg is available
        if not check_ffmpeg():
            raise ValueError("FFmpeg is required for video compression")
        
        # Get video info
        video_info = get_video_info(input_path)
        
        # Determine output dimensions
        if width is None and height is None:
            # Use original dimensions
            width = video_info['width']
            height = video_info['height']
        elif width is None:
            # Calculate width to maintain aspect ratio
            aspect_ratio = video_info['width'] / video_info['height']
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height to maintain aspect ratio
            aspect_ratio = video_info['width'] / video_info['height']
            height = int(width / aspect_ratio)
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-y',
            '-i', input_path,
            '-c:v', 'libx264',
            '-crf', str(quality),
            '-preset', 'medium',
            '-vf', f'scale={width}:{height}',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        
        # Run FFmpeg command
        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        
        # Check output file size
        input_size = os.path.getsize(input_path)
        output_size = os.path.getsize(output_path)
        
        # Log compression results
        logger.info(f"Compressed video from {input_size / 1024 / 1024:.2f} MB to {output_size / 1024 / 1024:.2f} MB")
        logger.info(f"Compression ratio: {input_size / output_size:.2f}x")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Error compressing video: {e}")
        raise


def calculate_frame_difference(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """
    Calculate difference between two frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        
    Returns:
        float: Mean absolute difference
    """
    # Ensure frames have same shape
    if frame1.shape != frame2.shape:
        raise ValueError(f"Frame shapes do not match: {frame1.shape} vs {frame2.shape}")
    
    # Calculate mean absolute difference
    return np.mean(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))


def detect_scene_changes(video_path: str, threshold: float = 30.0) -> List[int]:
    """
    Detect scene changes in video.
    
    Args:
        video_path: Path to video file
        threshold: Difference threshold for scene change detection
        
    Returns:
        list: Frame indices of scene changes
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Initialize variables
        prev_frame = None
        scene_changes = []
        frame_idx = 0
        
        # Process frames
        with tqdm(desc="Detecting scene changes") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for more stable comparison
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Compare with previous frame
                if prev_frame is not None:
                    # Calculate difference
                    diff = calculate_frame_difference(gray, prev_frame)
                    
                    # Check if difference exceeds threshold
                    if diff > threshold:
                        scene_changes.append(frame_idx)
                
                # Update previous frame
                prev_frame = gray
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        
        return scene_changes
    
    except Exception as e:
        logger.error(f"Error detecting scene changes: {e}")
        raise


def extract_keyframes(video_path: str, output_dir: str, method: str = 'scene_change',
                     max_frames: Optional[int] = None) -> List[str]:
    """
    Extract key frames from video.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save key frames
        method: Method for key frame extraction ('scene_change', 'uniform', 'content')
        max_frames: Maximum number of key frames to extract
        
    Returns:
        list: Paths to extracted key frames
    """
    try:
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video info
        video_info = get_video_info(video_path)
        total_frames = video_info['total_frames']
        
        # Determine key frame indices based on method
        if method == 'scene_change':
            # Detect scene changes
            key_indices = detect_scene_changes(video_path)
            
            # Add first and last frames
            if 0 not in key_indices:
                key_indices.insert(0, 0)
            if total_frames - 1 not in key_indices:
                key_indices.append(total_frames - 1)
                
        elif method == 'uniform':
            # Determine number of frames to extract
            if max_frames is None:
                max_frames = min(10, total_frames)
            
            # Extract frames at uniform intervals
            step = total_frames / max_frames
            key_indices = [int(i * step) for i in range(max_frames)]
            
        elif method == 'content':
            # This would require content analysis - use scene change as fallback
            key_indices = detect_scene_changes(video_path)
            
            # Add first and last frames
            if 0 not in key_indices:
                key_indices.insert(0, 0)
            if total_frames - 1 not in key_indices:
                key_indices.append(total_frames - 1)
                
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Limit number of key frames if specified
        if max_frames is not None and len(key_indices) > max_frames:
            # Keep first, last, and evenly spaced frames in between
            first_idx = key_indices[0]
            last_idx = key_indices[-1]
            
            middle_indices = key_indices[1:-1]
            selected_middle = []
            
            if max_frames > 2:
                step = len(middle_indices) / (max_frames - 2)
                selected_middle = [middle_indices[int(i * step)] for i in range(max_frames - 2)]
            
            key_indices = [first_idx] + selected_middle + [last_idx]
        
        # Extract frames
        return extract_frames(video_path, output_dir, key_indices)
    
    except Exception as e:
        logger.error(f"Error extracting key frames: {e}")
        raise