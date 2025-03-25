#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

install_requires = [
    # Core dependencies
    'numpy>=1.19.0',
    'scipy>=1.5.0',
    'tqdm>=4.48.0',
    'matplotlib>=3.3.0',
    'pillow>=8.0.0',
    
    # Deep learning
    'torch>=1.7.0',
    'torchvision>=0.8.0',
    'torchaudio>=0.7.0',
    'transformers>=4.5.0',
    
    # Image processing
    'opencv-python>=4.5.0',
    'scikit-image>=0.18.0',
    
    # Audio processing
    'librosa>=0.8.0',
    'pydub>=0.25.0',
    
    # Text processing
    'nltk>=3.6.0',
    'textblob>=0.15.0',
    
    # Video processing
    'moviepy>=1.0.0',
    
    # Error correction
    'reedsolo>=1.5.0',
    
    # Cryptography
    'cryptography>=3.4.0',
    
    # Command-line interface
    'click>=8.0.0',
    'colorama>=0.4.4',
]

setup_requires = [
    'pytest-runner>=5.3.0',
]

tests_require = [
    'pytest>=6.2.0',
    'pytest-cov>=2.12.0',
]

dev_requires = [
    # Linting and formatting
    'flake8>=3.9.0',
    'black>=21.5b0',
    'isort>=5.9.0',
    
    # Type checking
    'mypy>=0.812',
    
    # Documentation
    'sphinx>=4.0.0',
    'sphinx-rtd-theme>=0.5.0',
    
    # Packaging
    'twine>=3.4.0',
    'wheel>=0.36.0',
    'bumpversion>=0.6.0',
]

extras_require = {
    'dev': dev_requires + tests_require,
    'test': tests_require,
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'sphinx-autodoc-typehints>=1.12.0',
    ],
    'image': [
        'opencv-python>=4.5.0',
        'scikit-image>=0.18.0',
    ],
    'audio': [
        'librosa>=0.8.0',
        'pydub>=0.25.0',
    ],
    'text': [
        'nltk>=3.6.0',
        'textblob>=0.15.0',
    ],
    'video': [
        'moviepy>=1.0.0',
    ],
    'all': [
        'opencv-python>=4.5.0',
        'scikit-image>=0.18.0',
        'librosa>=0.8.0',
        'pydub>=0.25.0',
        'nltk>=3.6.0',
        'textblob>=0.15.0',
        'moviepy>=1.0.0',
    ],
}

setup(
    name='stegoai',
    version='0.2.0',
    description='Comprehensive steganography toolkit using deep learning',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Azhar',
    author_email='skazharuddin2003@gmail.com',
    url='https://github.com/azhar/stego-ai',
    packages=find_packages(include=['stegoai', 'stegoai.*']),
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'stegoai=stegoai.cli:main',
        ],
    },
    license='MIT',
    zip_safe=False,
    keywords=[
        'steganography',
        'deep learning',
        'data hiding',
        'security',
        'cryptography',
        'image steganography',
        'audio steganography',
        'text steganography',
        'video steganography',
        'network steganography',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security :: Cryptography',
    ],
    python_requires='>=3.7',
)