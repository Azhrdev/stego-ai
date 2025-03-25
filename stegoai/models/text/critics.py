# -*- coding: utf-8 -*-
"""
Text steganography critics for Stego-AI.

This module implements critic networks for detecting hidden messages
in text using different steganographic techniques.
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Union, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from stegoai.models.base import BaseCritic
from stegoai.utils.text_utils import find_zero_width_characters, strip_zero_width_characters

# Set up logging
logger = logging.getLogger(__name__)

# Zero-width characters used in whitespace steganography
ZERO_WIDTH_CHARS = {
    '0': '\u200B',  # Zero width space
    '1': '\u200C',  # Zero width non-joiner
    'start': '\u200D',  # Zero width joiner (used as start marker)
    'end': '\u2060',  # Word joiner (used as end marker)
}


class TextWhitespaceCritic(BaseCritic):
    """
    Critic for detecting messages hidden using zero-width characters.
    
    This critic analyzes text to identify patterns of zero-width characters
    that are characteristic of the whitespace steganography method.
    """
    
    def __init__(self):
        """Initialize the whitespace critic."""
        super().__init__()
        
        # Define detection thresholds
        self.thresholds = {
            'zwc_count': 10,  # Minimum number of zero-width characters for suspicion
            'pattern_score': 0.7,  # Threshold for suspicion based on pattern analysis
        }
    
    def forward(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze text for hidden messages using zero-width characters.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Find zero-width characters
        zero_width_chars = find_zero_width_characters(text)
        zwc_count = len(zero_width_chars)
        
        # If no zero-width characters, likely clean
        if zwc_count == 0:
            return {
                'probability': 0.0,
                'assessment': "Likely clean",
                'confidence': "High",
                'details': {
                    'zwc_count': 0,
                    'pattern_score': 0.0,
                }
            }
        
        # Check for start and end markers
        chars = ''.join(char for _, char in zero_width_chars)
        start_marker = ZERO_WIDTH_CHARS['start']
        end_marker = ZERO_WIDTH_CHARS['end']
        
        start_idx = chars.find(start_marker)
        end_idx = chars.find(end_marker, start_idx + 1) if start_idx != -1 else -1
        
        # Calculate pattern score
        pattern_score = 0.0
        
        # Check for contiguous placement
        is_contiguous = all(zero_width_chars[i+1][0] - zero_width_chars[i][0] == 1 
                            for i in range(len(zero_width_chars) - 1))
        
        # Count different types of zero-width chars
        char_types = set(char for _, char in zero_width_chars)
        has_diverse_chars = len(char_types) >= 2
        
        # Calculate pattern score
        if start_idx != -1 and end_idx != -1:
            # Strong pattern: both markers present
            pattern_score = 0.95
        elif start_idx != -1 or end_idx != -1:
            # Medium pattern: one marker present
            pattern_score = 0.8
        elif is_contiguous and has_diverse_chars:
            # Medium pattern: contiguous placement of diverse chars
            pattern_score = 0.7
        elif has_diverse_chars:
            # Weak pattern: diverse chars but not contiguous
            pattern_score = 0.5
        elif is_contiguous:
            # Weak pattern: contiguous but same char
            pattern_score = 0.4
        elif zwc_count > self.thresholds['zwc_count']:
            # Suspicious due to count
            pattern_score = 0.3
        else:
            # Weak pattern
            pattern_score = 0.2
        
        # Calculate overall probability
        if pattern_score >= 0.8:
            probability = pattern_score
            assessment = "Likely contains hidden data"
            confidence = "High"
        elif pattern_score >= 0.5:
            probability = pattern_score
            assessment = "Possibly contains hidden data"
            confidence = "Medium"
        else:
            probability = pattern_score
            assessment = "Possibly clean"
            confidence = "Low"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'zwc_count': zwc_count,
                'pattern_score': pattern_score,
                'has_start_marker': start_idx != -1,
                'has_end_marker': end_idx != -1,
            }
        }


class TextSynonymCritic(BaseCritic):
    """
    Critic for detecting messages hidden using synonym substitution.
    
    This critic analyzes text for unusual patterns of word choice
    that might indicate synonym-based steganography.
    """
    
    def __init__(self, synonym_dict: Optional[Dict[str, List[str]]] = None):
        """
        Initialize the synonym critic.
        
        Args:
            synonym_dict: Dictionary of words to their synonyms
        """
        super().__init__()
        self.synonym_dict = synonym_dict or self._load_default_synonyms()
        
        # Define detection thresholds
        self.thresholds = {
            'synonym_ratio': 0.2,  # Threshold for unusual synonym usage
            'rare_synonym_ratio': 0.1,  # Threshold for rare synonym usage
        }
    
    def forward(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze text for hidden messages using synonym substitution.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Extract words from text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Skip short texts
        if len(words) < 20:
            return {
                'probability': 0.1,
                'assessment': "Insufficient text for analysis",
                'confidence': "Low",
                'details': {
                    'word_count': len(words),
                    'synonym_ratio': 0.0,
                    'rare_synonym_ratio': 0.0,
                }
            }
        
        # Count words that have synonyms
        synonym_words = 0
        rare_synonyms = 0
        
        for word in words:
            if word in self.synonym_dict and len(self.synonym_dict[word]) > 1:
                synonym_words += 1
                
                # Check if it's a rare synonym
                synonyms = self.synonym_dict[word]
                primary_synonym = synonyms[0]  # First synonym is considered primary
                
                if word != primary_synonym:
                    rare_synonyms += 1
        
        # Calculate ratios
        synonym_ratio = synonym_words / len(words) if words else 0
        rare_synonym_ratio = rare_synonyms / synonym_words if synonym_words else 0
        
        # Calculate probability and assessment
        if rare_synonym_ratio > self.thresholds['rare_synonym_ratio'] * 1.5:
            probability = min(0.9, rare_synonym_ratio * 1.2)
            assessment = "Likely contains hidden data"
            confidence = "Medium"
        elif rare_synonym_ratio > self.thresholds['rare_synonym_ratio']:
            probability = min(0.7, rare_synonym_ratio)
            assessment = "Possibly contains hidden data"
            confidence = "Medium"
        elif synonym_ratio > self.thresholds['synonym_ratio']:
            probability = synonym_ratio * 0.8
            assessment = "Unusual synonym usage detected"
            confidence = "Low"
        else:
            probability = 0.1
            assessment = "Likely clean"
            confidence = "Medium"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'word_count': len(words),
                'synonym_words': synonym_words,
                'rare_synonyms': rare_synonyms,
                'synonym_ratio': synonym_ratio,
                'rare_synonym_ratio': rare_synonym_ratio,
            }
        }
    
    def _load_default_synonyms(self) -> Dict[str, List[str]]:
        """
        Load a default synonym dictionary.
        
        Returns:
            dict: Mapping of words to synonyms
        """
        # Just use a basic dictionary for demonstration
        # In a real implementation, this would use a much larger dataset
        return {
            # Common adjectives
            'big': ['big', 'large', 'huge', 'enormous', 'massive', 'substantial'],
            'small': ['small', 'tiny', 'little', 'miniature', 'petite', 'minute'],
            'good': ['good', 'great', 'excellent', 'fine', 'superior', 'wonderful'],
            'bad': ['bad', 'poor', 'terrible', 'awful', 'dreadful', 'horrible'],
            'happy': ['happy', 'glad', 'joyful', 'cheerful', 'delighted', 'pleased'],
            'sad': ['sad', 'unhappy', 'sorrowful', 'dejected', 'depressed', 'gloomy'],
            
            # Common verbs
            'run': ['run', 'sprint', 'dash', 'jog', 'race', 'gallop'],
            'walk': ['walk', 'stroll', 'stride', 'saunter', 'amble', 'wander'],
            'look': ['look', 'see', 'view', 'observe', 'watch', 'examine'],
            'eat': ['eat', 'consume', 'devour', 'ingest', 'dine', 'feast'],
            'say': ['say', 'tell', 'speak', 'utter', 'express', 'articulate'],
            'think': ['think', 'believe', 'consider', 'contemplate', 'ponder', 'reflect'],
            
            # Common nouns
            'house': ['house', 'home', 'residence', 'dwelling', 'abode', 'domicile'],
            'car': ['car', 'vehicle', 'automobile', 'motor', 'machine', 'ride'],
            'job': ['job', 'work', 'occupation', 'profession', 'career', 'vocation'],
            'money': ['money', 'cash', 'currency', 'funds', 'wealth', 'capital'],
            'food': ['food', 'sustenance', 'nourishment', 'fare', 'cuisine', 'provisions'],
            'friend': ['friend', 'companion', 'pal', 'ally', 'associate', 'confidant'],
        }


class TextCapitalizationCritic(BaseCritic):
    """
    Critic for detecting messages hidden using capitalization patterns.
    
    This critic analyzes text for unusual capitalization patterns
    that might indicate capitalization-based steganography.
    """
    
    def __init__(self):
        """Initialize the capitalization critic."""
        super().__init__()
        
        # Define detection thresholds
        self.thresholds = {
            'inconsistent_ratio': 0.05,  # Threshold for inconsistent capitalization
        }
    
    def forward(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze text for hidden messages using capitalization patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Extract words with context
        words_with_context = self._extract_words_with_context(text)
        
        # Skip short texts
        if len(words_with_context) < 20:
            return {
                'probability': 0.1,
                'assessment': "Insufficient text for analysis",
                'confidence': "Low",
                'details': {
                    'word_count': len(words_with_context),
                    'inconsistent_count': 0,
                    'inconsistent_ratio': 0.0,
                }
            }
        
        # Count inconsistent capitalization
        inconsistent_caps = 0
        for word, is_sentence_start in words_with_context:
            # Skip short words
            if len(word) < 2:
                continue
            
            # Skip all-uppercase words (likely acronyms)
            if word.isupper():
                continue
            
            # Check for inconsistent capitalization
            if is_sentence_start and word[0].islower():
                # Sentence start should be capitalized
                inconsistent_caps += 1
            elif not is_sentence_start and word[0].isupper():
                # Check if it's a common word (not a proper noun)
                common_words = {'the', 'a', 'an', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 
                                'with', 'by', 'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under'}
                
                if word.lower() in common_words:
                    inconsistent_caps += 1
        
        # Calculate inconsistent ratio
        inconsistent_ratio = inconsistent_caps / len(words_with_context) if words_with_context else 0
        
        # Calculate probability and assessment
        if inconsistent_ratio > self.thresholds['inconsistent_ratio'] * 2:
            probability = min(0.9, inconsistent_ratio * 8)
            assessment = "Likely contains hidden data"
            confidence = "High"
        elif inconsistent_ratio > self.thresholds['inconsistent_ratio']:
            probability = min(0.7, inconsistent_ratio * 6)
            assessment = "Possibly contains hidden data"
            confidence = "Medium"
        else:
            probability = max(0.1, inconsistent_ratio * 4)
            assessment = "Likely clean"
            confidence = "Medium"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'word_count': len(words_with_context),
                'inconsistent_count': inconsistent_caps,
                'inconsistent_ratio': inconsistent_ratio,
            }
        }
    
    def _extract_words_with_context(self, text: str) -> List[Tuple[str, bool]]:
        """
        Extract words from text with context about sentence boundaries.
        
        Args:
            text: Text to analyze
            
        Returns:
            list: Tuples of (word, is_sentence_start)
        """
        words_with_context = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence)
            
            for i, word in enumerate(words):
                is_sentence_start = (i == 0)
                words_with_context.append((word, is_sentence_start))
        
        return words_with_context


class TextGenerativeCritic(BaseCritic):
    """
    Critic for detecting messages hidden using language model generation patterns.
    
    This critic analyzes text for patterns that indicate the presence of
    special words or sequences used in generative steganography.
    """
    
    def __init__(self):
        """Initialize the generative critic."""
        super().__init__()
        
        # Define detection thresholds
        self.thresholds = {
            'marker_words_ratio': 0.05,  # Threshold for marker words
            'coherence_drop': 0.3,       # Threshold for coherence drop
        }
        
        # Words known to be used in encoding
        self.marker_words = {
            'always', 'certainly', 'definitely',
            'sometimes', 'occasionally', 'periodically',
            'rarely', 'seldom', 'hardly',
            'never', 'none', 'nil',
            'confidently', 'probably', 'likely',
            'possibly', 'perhaps', 'maybe',
            'unlikely', 'doubtfully', 'questionably',
            'infrequently', 'barely', 'scarcely',
            'almost', 'nearly', 'virtually',
            'nowhere', 'nought'
        }
        
        # Phrases that might indicate a summary section (where encoded data often resides)
        self.summary_phrases = [
            "in summary", "in conclusion", "to summarize", "to conclude",
            "in the end", "finally", "lastly", "to sum up"
        ]
    
    def forward(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Analyze text for hidden messages using generative patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Analysis results containing probability, assessment, and confidence
        """
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Skip short texts
        if len(words) < 30:
            return {
                'probability': 0.1,
                'assessment': "Insufficient text for analysis",
                'confidence': "Low",
                'details': {
                    'word_count': len(words),
                    'marker_words_count': 0,
                    'marker_words_ratio': 0.0,
                    'has_summary_section': False,
                }
            }
        
        # Check for summary section
        has_summary_section = False
        summary_start_idx = len(words)
        
        for phrase in self.summary_phrases:
            idx = text.lower().find(phrase)
            if idx != -1:
                has_summary_section = True
                
                # Find the word index for this position
                prefix_words = len(re.findall(r'\b\w+\b', text.lower()[:idx]))
                summary_start_idx = min(summary_start_idx, prefix_words)
        
        # Count marker words
        marker_words_count = 0
        marker_words_in_summary = 0
        
        for i, word in enumerate(words):
            if word in self.marker_words:
                marker_words_count += 1
                if i >= summary_start_idx:
                    marker_words_in_summary += 1
        
        # Calculate marker words ratio
        marker_words_ratio = marker_words_count / len(words) if words else 0
        
        # Calculate ratio in summary section
        summary_words = len(words) - summary_start_idx if has_summary_section else 0
        summary_ratio = marker_words_in_summary / summary_words if summary_words else 0
        
        # Check for coherence drop (simplified)
        coherence_drop = 0.0
        if has_summary_section and summary_words > 10:
            # In a real implementation, this would use a language model to measure
            # how much the coherence/perplexity changes in the summary section
            
            # For now, use a heuristic based on marker word density
            main_ratio = (marker_words_count - marker_words_in_summary) / summary_start_idx if summary_start_idx else 0
            coherence_drop = max(0, summary_ratio - main_ratio * 2)
        
        # Calculate probability and assessment
        if has_summary_section and summary_ratio > self.thresholds['marker_words_ratio'] * 3:
            probability = min(0.9, summary_ratio * 2)
            assessment = "Likely contains hidden data"
            confidence = "High"
        elif coherence_drop > self.thresholds['coherence_drop']:
            probability = min(0.8, coherence_drop * 1.5)
            assessment = "Likely contains hidden data"
            confidence = "Medium"
        elif marker_words_ratio > self.thresholds['marker_words_ratio'] * 2:
            probability = min(0.7, marker_words_ratio * 5)
            assessment = "Possibly contains hidden data"
            confidence = "Medium"
        elif has_summary_section and summary_ratio > self.thresholds['marker_words_ratio']:
            probability = min(0.6, summary_ratio * 1.5)
            assessment = "Possibly contains hidden data"
            confidence = "Low"
        else:
            probability = max(0.1, marker_words_ratio * 3)
            assessment = "Likely clean"
            confidence = "Medium"
        
        return {
            'probability': probability,
            'assessment': assessment,
            'confidence': confidence,
            'details': {
                'word_count': len(words),
                'marker_words_count': marker_words_count,
                'marker_words_ratio': marker_words_ratio,
                'has_summary_section': has_summary_section,
                'summary_ratio': summary_ratio if has_summary_section else 0.0,
                'coherence_drop': coherence_drop,
            }
        }


def get_text_critic(method: str = 'whitespace', **kwargs) -> BaseCritic:
    """
    Factory function to get the appropriate text critic.
    
    Args:
        method: Steganography method ('whitespace', 'synonym', 'capitalization', 'generative')
        **kwargs: Additional arguments to pass to the critic
        
    Returns:
        BaseCritic: Appropriate critic for the method
        
    Raises:
        ValueError: If method is not supported
    """
    if method == 'whitespace':
        return TextWhitespaceCritic(**kwargs)
    elif method == 'synonym':
        return TextSynonymCritic(**kwargs)
    elif method == 'capitalization':
        return TextCapitalizationCritic(**kwargs)
    elif method == 'generative':
        return TextGenerativeCritic(**kwargs)
    else:
        raise ValueError(f"Unsupported text steganography method: {method}")