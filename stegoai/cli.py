#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line interface for Stego-AI.

This module provides a comprehensive command-line interface for using the
steganography capabilities of Stego-AI across different media types.
"""

import os
import sys
import logging
import click
from tqdm import tqdm

from stegoai import get_model
from stegoai.utils.image_utils import get_image_dimensions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stegoai-cli')


class CLIProgressCallback:
    """Callback for progress updates in the CLI."""
    
    def __init__(self, desc="Processing"):
        self.progress_bar = tqdm(total=100, desc=desc)
        self.last_value = 0
        
    def __call__(self, progress, info=None):
        """Update progress bar."""
        current = int(progress * 100)
        increment = current - self.last_value
        if increment > 0:
            self.progress_bar.update(increment)
            self.last_value = current
            if info:
                self.progress_bar.set_postfix_str(info)
                
    def close(self):
        """Close progress bar."""
        self.progress_bar.close()


@click.group()
def cli():
    """Stego-AI: Deep learning-based steganography toolkit."""
    pass


# Image steganography commands
@cli.group()
def image():
    """Image steganography commands."""
    pass


@image.command('encode')
@click.argument('cover', type=click.Path(exists=True))
@click.argument('message', type=str)
@click.option('-o', '--output', type=click.Path(), help='Output stego image path')
@click.option('-a', '--architecture', type=str, default='dense', 
              help='Model architecture (simple, residual, dense, unet)')
@click.option('-m', '--multiple', is_flag=True, help='Encode message multiple times to fill capacity')
@click.option('-q', '--quality', type=int, default=95, help='JPEG quality (0-100)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def image_encode(cover, message, output, architecture, multiple, quality, verbose):
    """Hide a message in an image."""
    if not output:
        output = f'stego_{os.path.basename(cover)}'
    
    try:
        # Load the model
        model = get_model('image', architecture=architecture, verbose=verbose)
        
        click.echo(f"Encoding message into {cover}...")
        model.encode(cover, output, message, multiple, quality)
        click.echo(f"Message successfully encoded to {output}")
        
        # Show capacity usage
        if verbose:
            width, height = get_image_dimensions(cover)
            capacity = width * height * model.data_depth
            msg_size = len(message) * 8  # rough estimate
            click.echo(f"Capacity: {capacity} bits, Message: ~{msg_size} bits, "
                      f"Usage: ~{msg_size/capacity*100:.1f}%")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@image.command('decode')
@click.argument('stego', type=click.Path(exists=True))
@click.option('-a', '--architecture', type=str, default='dense',
              help='Model architecture (simple, residual, dense, unet)')
@click.option('-l', '--max-length', type=int, help='Maximum message length in bits')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def image_decode(stego, architecture, max_length, verbose):
    """Extract a hidden message from an image."""
    try:
        # Load the model
        model = get_model('image', architecture=architecture, verbose=verbose)
        
        click.echo(f"Decoding message from {stego}...")
        message = model.decode(stego, max_length)
        click.echo("\nExtracted message:")
        click.echo(f"\n{message}\n")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@image.command('analyze')
@click.argument('image', type=click.Path(exists=True))
@click.option('-a', '--architecture', type=str, default='dense',
              help='Model architecture (simple, residual, dense, unet)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def image_analyze(image, architecture, verbose):
    """Analyze an image for steganographic content."""
    try:
        # Load the model
        model = get_model('image', architecture=architecture, verbose=verbose)
        
        click.echo(f"Analyzing {image}...")
        result = model.analyze_image(image)
        
        click.echo("\nAnalysis results:")
        click.echo(f"Assessment: {result['assessment']}")
        click.echo(f"Confidence: {result['confidence']}")
        click.echo(f"Probability: {result['probability']:.2f}")
        click.echo(f"Score: {result['score']:.2f}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@image.command('capacity')
@click.argument('image', type=click.Path(exists=True))
@click.option('-a', '--architecture', type=str, default='dense',
              help='Model architecture (simple, residual, dense, unet)')
def image_capacity(image, architecture):
    """Estimate capacity of an image for steganography."""
    try:
        # Get image dimensions
        width, height = get_image_dimensions(image)
        
        # Create model to get data_depth
        model = get_model('image', architecture=architecture)
        
        # Calculate capacity
        capacity_bits = width * height * model.data_depth
        capacity_bytes = capacity_bits // 8
        capacity_chars = capacity_bytes  # Approximation for ASCII
        
        # Calculate estimated capacities for different content
        click.echo("\nEstimated capacity:")
        click.echo(f"Dimensions: {width}x{height} pixels")
        click.echo(f"Architecture: {architecture} ({model.data_depth} bits per pixel)")
        click.echo(f"Raw capacity: {capacity_bits:,} bits ({capacity_bytes:,} bytes)")
        click.echo(f"Text: ~{capacity_chars:,} characters")
        click.echo(f"Base64 data: ~{capacity_bytes // 4 * 3:,} bytes")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@image.command('compare')
@click.argument('original', type=click.Path(exists=True))
@click.argument('modified', type=click.Path(exists=True))
@click.option('-a', '--architecture', type=str, default='dense',
              help='Model architecture (simple, residual, dense, unet)')
def image_compare(original, modified, architecture):
    """Compare original and potentially steganographic images."""
    try:
        # Load the model
        model = get_model('image', architecture=architecture)
        
        click.echo(f"Comparing {original} and {modified}...")
        result = model.compare_images(original, modified)
        
        click.echo("\nComparison results:")
        click.echo(f"Quality assessment: {result['quality_assessment']}")
        click.echo(f"PSNR: {result['psnr']:.2f} dB")
        click.echo(f"SSIM: {result['ssim']:.4f}")
        click.echo(f"MSE: {result['mse']:.6f}")
        click.echo(f"Mean difference: {result['mean_diff']:.2f}")
        click.echo(f"Maximum difference: {result['max_diff']:.2f}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Audio steganography commands
@cli.group()
def audio():
    """Audio steganography commands."""
    pass


@audio.command('encode')
@click.argument('cover', type=click.Path(exists=True))
@click.argument('message', type=str)
@click.option('-o', '--output', type=click.Path(), help='Output stego audio path')
@click.option('-m', '--method', type=str, default='spectrogram',
              help='Method (spectrogram, waveform, phase)')
@click.option('-q', '--quality', type=click.Choice(['low', 'medium', 'high']), 
              default='high', help='Audio quality')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def audio_encode(cover, message, output, method, quality, verbose):
    """Hide a message in an audio file."""
    if not output:
        output = f'stego_{os.path.basename(cover)}'
    
    try:
        # Load the model
        model = get_model('audio', mode=method, verbose=verbose)
        
        click.echo(f"Encoding message into {cover}...")
        model.encode(cover, output, message, quality)
        click.echo(f"Message successfully encoded to {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@audio.command('decode')
@click.argument('stego', type=click.Path(exists=True))
@click.option('-m', '--method', type=str, default='spectrogram',
              help='Method (spectrogram, waveform, phase)')
@click.option('-l', '--max-length', type=int, help='Maximum message length in bits')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def audio_decode(stego, method, max_length, verbose):
    """Extract a hidden message from an audio file."""
    try:
        # Load the model
        model = get_model('audio', mode=method, verbose=verbose)
        
        click.echo(f"Decoding message from {stego}...")
        message = model.decode(stego, max_length)
        click.echo("\nExtracted message:")
        click.echo(f"\n{message}\n")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@audio.command('analyze')
@click.argument('audio', type=click.Path(exists=True))
@click.option('-m', '--method', type=str, default='spectrogram',
              help='Method (spectrogram, waveform, phase)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def audio_analyze(audio, method, verbose):
    """Analyze an audio file for steganographic content."""
    try:
        # Load the model
        model = get_model('audio', mode=method, verbose=verbose)
        
        click.echo(f"Analyzing {audio}...")
        result = model.analyze_audio(audio)
        
        click.echo("\nAnalysis results:")
        click.echo(f"Assessment: {result['assessment']}")
        click.echo(f"Confidence: {result['confidence']}")
        click.echo(f"Probability: {result['probability']:.2f}")
        click.echo(f"Score: {result['score']:.2f}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Text steganography commands
@cli.group()
def text():
    """Text steganography commands."""
    pass


@text.command('encode')
@click.argument('cover', type=click.Path(exists=True))
@click.argument('message', type=str)
@click.option('-o', '--output', type=click.Path(), help='Output stego text path')
@click.option('-m', '--method', type=str, default='whitespace',
              help='Method (whitespace, synonym, capitalization, generative)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def text_encode(cover, message, output, method, verbose):
    """Hide a message in a text file."""
    if not output:
        output = f'stego_{os.path.basename(cover)}'
    
    try:
        # Load the model
        model = get_model('text', method=method, verbose=verbose)
        
        click.echo(f"Encoding message into {cover}...")
        model.encode(cover, output, message)
        click.echo(f"Message successfully encoded to {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@text.command('decode')
@click.argument('stego', type=click.Path(exists=True))
@click.option('-m', '--method', type=str, default='whitespace',
              help='Method (whitespace, synonym, capitalization, generative)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def text_decode(stego, method, verbose):
    """Extract a hidden message from a text file."""
    try:
        # Load the model
        model = get_model('text', method=method, verbose=verbose)
        
        click.echo(f"Decoding message from {stego}...")
        message = model.decode(stego)
        click.echo("\nExtracted message:")
        click.echo(f"\n{message}\n")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@text.command('analyze')
@click.argument('text_file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def text_analyze(text_file, verbose):
    """Analyze a text file for steganographic content."""
    try:
        click.echo(f"Analyzing {text_file}...")
        
        # Create models for each method
        methods = ['whitespace', 'synonym', 'capitalization', 'generative']
        results = {}
        
        for method in methods:
            model = get_model('text', method=method, verbose=False)
            results[method] = model.analyze_text(text_file)
        
        click.echo("\nAnalysis results:")
        for method, result in results.items():
            click.echo(f"\n{method.capitalize()} method:")
            click.echo(f"  Assessment: {result['assessment']}")
            click.echo(f"  Confidence: {result['confidence']}")
            click.echo(f"  Probability: {result['probability']:.2f}")
        
        # Overall assessment
        max_prob = max(result['probability'] for result in results.values())
        overall = "Likely contains hidden data" if max_prob > 0.7 else "Likely clean"
        click.echo(f"\nOverall assessment: {overall}")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Video steganography commands
@cli.group()
def video():
    """Video steganography commands."""
    pass


@video.command('encode')
@click.argument('cover', type=click.Path(exists=True))
@click.argument('message', type=str)
@click.option('-o', '--output', type=click.Path(), help='Output stego video path')
@click.option('-m', '--method', type=str, default='frame_lsb',
              help='Method (frame_lsb, frame_dct, temporal, neural)')
@click.option('-q', '--quality', type=int, default=23, help='Video quality (CRF value, lower is better)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def video_encode(cover, message, output, method, quality, verbose):
    """Hide a message in a video file."""
    if not output:
        output = f'stego_{os.path.basename(cover)}'
    
    try:
        # Load the model
        model = get_model('video', method=method, verbose=verbose)
        
        click.echo(f"Encoding message into {cover}...")
        model.encode(cover, output, message, quality)
        click.echo(f"Message successfully encoded to {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@video.command('decode')
@click.argument('stego', type=click.Path(exists=True))
@click.option('-m', '--method', type=str, default='frame_lsb',
              help='Method (frame_lsb, frame_dct, temporal, neural)')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def video_decode(stego, method, verbose):
    """Extract a hidden message from a video file."""
    try:
        # Load the model
        model = get_model('video', method=method, verbose=verbose)
        
        click.echo(f"Decoding message from {stego}...")
        message = model.decode(stego)
        click.echo("\nExtracted message:")
        click.echo(f"\n{message}\n")
    
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Network steganography commands
@cli.group()
def network():
    """Network steganography commands."""
    pass


@network.command('encode')
@click.argument('target')
@click.argument('message', type=str)
@click.option('-m', '--method', type=str, default='header',
              help='Method (header, timing, size, sequence, covert_channel)')
@click.option('-r', '--protocol', type=str, default='tcp',
              help='Protocol (tcp, udp, icmp, dns)')
@click.option('-p', '--port', type=int, default=8000, help='Target port')
@click.option('-i', '--interface', type=str, default='eth0', help='Network interface')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def network_encode(target, message, method, protocol, port, interface, verbose):
    """Send a message steganographically over the network."""
    try:
        # Load the model
        model = get_model('network', method=method, protocol=protocol,
                         interface=interface, port=port, verbose=verbose)
        
        click.echo(f"Sending message to {target}...")
        success = model.encode(target, message, port)
        
        if success:
            click.echo("Message sent successfully")
        else:
            click.echo("Failed to send message", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@network.command('decode')
@click.option('-m', '--method', type=str, default='header',
              help='Method (header, timing, size, sequence, covert_channel)')
@click.option('-r', '--protocol', type=str, default='tcp',
              help='Protocol (tcp, udp, icmp, dns)')
@click.option('-p', '--port', type=int, default=8000, help='Listening port')
@click.option('-i', '--interface', type=str, default='eth0', help='Network interface')
@click.option('-t', '--time', type=int, default=60, help='Listen time in seconds')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
def network_decode(method, protocol, port, interface, time, verbose):
    """Listen for and extract steganographic messages from network traffic."""
    try:
        # Load the model
        model = get_model('network', method=method, protocol=protocol,
                         interface=interface, port=port, verbose=verbose)
        
        # Set up progress callback
        callback = CLIProgressCallback(desc="Listening")
        
        click.echo(f"Listening on {interface}:{port} for {time} seconds...")
        message = model.decode(time, callback)
        callback.close()
        
        click.echo("\nExtracted message:")
        click.echo(f"\n{message}\n")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@network.command('capacity')
@click.option('-m', '--method', type=str, default='header',
              help='Method (header, timing, size, sequence, covert_channel)')
@click.option('-r', '--protocol', type=str, default='tcp',
              help='Protocol (tcp, udp, icmp, dns)')
def network_capacity(method, protocol):
    """Estimate steganographic capacity of network methods."""
    try:
        # Create temporary model to use estimate_capacity
        model = get_model('network', method=method, protocol=protocol)
        
        # Get capacity estimates
        estimates = model.estimate_capacity(protocol, method)
        
        click.echo("\nCapacity estimates:")
        click.echo(f"Protocol: {estimates['protocol']}")
        click.echo(f"Method: {estimates['method']}")
        click.echo(f"Capacity per packet: {estimates['capacity_per_packet_bits']} bits")
        click.echo(f"Estimated packets per second: {estimates['estimated_packets_per_second']}")
        click.echo(f"Estimated throughput: {estimates['estimated_bits_per_second']} bits/second")
        click.echo(f"Estimated text capacity: {estimates['estimated_chars_per_second']} chars/second")
        click.echo(f"Estimated chars in 1 minute: {estimates['estimated_capacity_1min_chars']}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# General commands
@cli.command('version')
def version():
    """Show Stego-AI version."""
    import stegoai
    click.echo(f"Stego-AI version {stegoai.__version__}")


@cli.command('list')
def list_models():
    """List available steganography models."""
    click.echo("\nAvailable steganography models:")
    
    click.echo("\nImage steganography:")
    click.echo("  - simple: Basic model with good balance of speed and capacity")
    click.echo("  - residual: Better image quality, slightly slower")
    click.echo("  - dense: Highest capacity, best image quality, slowest")
    click.echo("  - unet: Excellent perceptual quality with good capacity")
    
    click.echo("\nAudio steganography:")
    click.echo("  - spectrogram: Hide data in frequency domain, good for music")
    click.echo("  - waveform: Direct time-domain hiding, better for speech")
    click.echo("  - phase: Modify phase information, less perceptible")
    
    click.echo("\nText steganography:")
    click.echo("  - whitespace: Use invisible spaces and formatting")
    click.echo("  - synonym: Replace words with synonyms based on bits")
    click.echo("  - capitalization: Modify capitalization patterns")
    click.echo("  - generative: Use language models to encode in word choice")
    
    click.echo("\nVideo steganography:")
    click.echo("  - frame_lsb: Modify least significant bits in select frames")
    click.echo("  - frame_dct: Hide in DCT coefficients like JPEG")
    click.echo("  - temporal: Use patterns across frames")
    click.echo("  - neural: Apply image models to keyframes")
    
    click.echo("\nNetwork steganography:")
    click.echo("  - header: Modify packet headers")
    click.echo("  - timing: Encode in packet timing")
    click.echo("  - size: Vary packet sizes to encode data")
    click.echo("  - sequence: Use packet sequence patterns")
    click.echo("  - covert_channel: Create hidden protocols")


def main():
    """Main entry point for CLI."""
    try:
        cli()
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()