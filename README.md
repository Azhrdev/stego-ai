# Stego-AI: Comprehensive Deep Learning-Based Steganography

[![License](https://img.shields.io/github/license/azhar/stego-ai.svg)](https://github.com/azhar/stego-ai/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyPI Shield](https://img.shields.io/pypi/v/stegoai.svg)](https://pypi.python.org/pypi/stegoai)

## Overview

**Stego-AI** is a comprehensive steganography toolkit that uses deep learning to hide messages in multiple media types. Unlike traditional steganography techniques, Stego-AI leverages neural networks to embed information in ways that are:

- **Difficult to detect**: Messages are hidden in ways that evade statistical analysis
- **Robust**: Messages can often be recovered even after some manipulation
- **High capacity**: Can hide more information than traditional methods
- **Multi-modal**: Supports hiding data in images, audio, text, video, and network traffic

This project is built with PyTorch and provides both a robust command-line interface and a Python API.

## Media Types

Stego-AI supports five different media types for steganography:

### 🖼️ Image Steganography
- Uses neural networks to hide data in image pixels
- Multiple architectures: simple, residual, dense, U-Net
- Excellent for hiding substantial amounts of data with minimal visual impact

### 🔊 Audio Steganography
- Hide data in audio spectrograms, waveforms, or phase information
- Maintains audio quality while embedding significant data
- Resistant to basic audio processing operations

### 📝 Text Steganography
- Multiple techniques: whitespace, synonym substitution, capitalization patterns
- Language model-based methods for generating natural-looking text with hidden messages
- Works across multiple languages

### 🎬 Video Steganography
- Frame-based approaches for high-capacity data hiding
- Temporal patterns for more robust hiding
- DCT coefficient manipulation for resistance to compression

### 🌐 Network Steganography
- Hide data in packet headers, timing, sizes, or sequences
- Create covert channels within legitimate traffic
- Subtle enough to evade basic network monitoring

## Installation

### From PyPI

```bash
pip install stegoai
```

### From Source

```bash
git clone https://github.com/azhar/stego-ai.git
cd stego-ai
pip install -e .
```

### With Extra Features

```bash
# Install with support for all media types
pip install stegoai[all]

# Or just specific media types
pip install stegoai[image,audio]
```

## Quick Start

### Command Line Interface

Hide a message in an image:
```bash
stegoai image encode cover.png -o stego.png "Your secret message"
```

Recover a hidden message:
```bash
stegoai image decode stego.png
```

Hide a message in audio:
```bash
stegoai audio encode -m spectrogram cover.wav -o stego.wav "Audio steganography is cool"
```

Hide a message in text:
```bash
stegoai text encode -m whitespace article.txt -o hidden.txt "Hidden message"
```

Hide a message in video:
```bash
stegoai video encode -m frame_lsb video.mp4 -o stego_video.mp4 "Video can hide data too"
```

Send a hidden message over the network:
```bash
stegoai network encode -m header -r tcp 192.168.1.100 "Network steganography in action"
```

### Python API

```python
from stegoai import get_model

# Load an image steganography model
model = get_model('image', architecture='dense')

# Hide a message
model.encode('cover.png', 'stego.png', 'This is a secret message!')

# Extract a hidden message
message = model.decode('stego.png')
print(message)  # 'This is a secret message!'
```

## Features

- **Deep Learning-Based**: Uses neural networks rather than traditional bit manipulation
- **Multi-Modal**: Support for images, audio, text, video, and network traffic
- **Adversarially Trained**: Many models use a discriminator network to minimize artifacts
- **Error Correction**: Ensures message integrity even with some degradation
- **Comprehensive CLI**: Unified interface for all media types
- **Analysis Tools**: Detect steganography and estimate capacity
- **Custom Models**: Train your own models with different architectures

## Development

To set up a development environment:

```bash
pip install -e ".[dev]"
```

Run tests:
```bash
pytest
```

## Documentation

For detailed documentation and examples, see the [official documentation](https://stego-ai.readthedocs.io/).

## Models and Methods

### Image Models
- **Simple**: Basic model with good balance of speed and capacity
- **Residual**: Better image quality, slightly slower
- **Dense**: Highest capacity, best image quality, slowest
- **U-Net**: Excellent perceptual quality with good capacity

### Audio Methods
- **Spectrogram**: Hide data in frequency domain, good for music
- **Waveform**: Direct time-domain hiding, better for speech
- **Phase**: Modify phase information, less perceptible

### Text Methods
- **Whitespace**: Use invisible spaces and formatting
- **Synonym**: Replace words with synonyms based on bits
- **Capitalization**: Modify capitalization patterns
- **Generative**: Use language models to encode in word choice

### Video Methods
- **Frame LSB**: Modify least significant bits in select frames
- **Frame DCT**: Hide in DCT coefficients like JPEG
- **Temporal**: Use patterns across frames
- **Neural**: Apply image models to keyframes

### Network Methods
- **Header**: Modify packet headers
- **Timing**: Encode in packet timing
- **Size**: Vary packet sizes to encode data
- **Sequence**: Use packet sequence patterns
- **Covert Channel**: Create hidden protocols

## Contributing

Contributions are welcome! Areas where help is particularly appreciated:

- Improved model architectures
- Support for additional media types or file formats
- Mobile platform support
- Enhanced detection resistance
- Better compression and error correction
- Web interface development

Please submit pull requests to the repository.

## Performance Benchmarks

### Image Steganography
| Model     | Capacity (bpp) | PSNR (dB) | Detection Resistance | Speed    |
|-----------|---------------|-----------|---------------------|----------|
| Simple    | 1-2           | 38-42     | Medium              | Fast     |
| Residual  | 2-3           | 40-45     | High                | Medium   |
| Dense     | 3-4           | 42-48     | Very High           | Slow     |
| U-Net     | 2-3           | 45-50     | Very High           | Medium   |

### Audio Steganography
| Method      | Capacity (bps) | SNR (dB) | Detection Resistance | Speech Quality |
|-------------|---------------|----------|---------------------|----------------|
| Spectrogram | 1000-2000     | 30-35    | High                | Good           |
| Waveform    | 500-1000      | 25-30    | Medium              | Very Good      |
| Phase       | 300-600       | 35-40    | Very High           | Excellent      |

## Security Considerations

Stego-AI provides strong steganography capabilities but is not a complete security solution on its own:

- For sensitive data, always use encryption before applying steganography
- All steganographic methods may be vulnerable to targeted steganalysis
- The security of the system depends on keeping the steganography method secret
- Consider operational security beyond the technical implementation

## Use Cases

- **Privacy-Preserving Communication**: Hide sensitive information in seemingly innocuous media
- **Digital Watermarking**: Embed ownership information in media
- **Covert Communication**: Create hidden channels in environments with monitoring
- **Data Exfiltration Protection**: Detect unauthorized data hiding
- **Educational Purposes**: Learn about steganography and data hiding techniques

## Citation

If you use Stego-AI in your research, please cite:

```
@software{stego-ai,
  author = {Azhar},
  title = {Stego-AI: Comprehensive Deep Learning Based Steganography},
  year = {2024},
  url = {https://github.com/azhar/stego-ai}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project draws inspiration from:
- SteganoGAN
- HiDDeN
- DeepSteg
- Various research papers on deep learning-based steganography

## FAQ

### How is this different from encryption?
Encryption scrambles data to make it unreadable without a key. Steganography hides the existence of the data itself. They can be used together for stronger security.

### What's the maximum message size I can hide?
It depends on the media type, model, and cover file size. Use the `capacity` command to estimate for your specific file:
```bash
stegoai image capacity cover.png
```

### Can I hide one type of media in another?
Yes! You can hide any data in any supported media type. For example, you could hide an audio file within an image or video.

### Is this detectable?
All steganography methods have some theoretical detection possibility. Stego-AI uses deep learning to minimize detectability, but it's not perfect. The neural methods tend to be more resistant to statistical steganalysis than traditional methods.

### Can I train custom models?
Absolutely! Stego-AI supports training custom models with your own data:
```bash
stegoai image train -a dense -o my_model.pt train_images/ validation_images/
```