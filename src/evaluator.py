"""
Evaluation framework for comparing VLM outputs with ground truth.

This module provides metrics and tools for assessing the quality of
VLM-generated transliterations and translations of Safaitic inscriptions.
"""

import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import unicodedata


class InscriptionEvaluator:
    """
    Evaluator for Safaitic inscription analysis results.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        self.safaitic_chars = set("ʾbgdhwzḥṭyklmnsʿfṣqrtṯḫḏḍġ¹²")
        
    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Convert to lowercase for case-insensitive comparison
        text = text.lower()
        
        return text.strip()
    
    @staticmethod
    def extract_transliteration(text: str) -> str:
        """
        Extract transliteration from VLM response that might contain extra text.
        
        Args:
            text: VLM response text
            
        Returns:
            Extracted transliteration
        """
        # Look for patterns like "transliteration: xyz" or just Safaitic characters
        patterns = [
            r'transliteration[:\s]+([^\n]+)',
            r'reading[:\s]+([^\n]+)',
            r'^l\s+\w+',  # Common Safaitic pattern starting with 'l'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return cleaned text
        return text.strip()
    
    def character_accuracy(
        self,
        ground_truth: str,
        prediction: str,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate character-level accuracy metrics.
        
        Args:
            ground_truth: Expected transliteration
            prediction: VLM-generated transliteration
            normalize: Whether to normalize texts before comparison
            
        Returns:
            Dictionary with accuracy metrics
        """
        if normalize:
            gt = self.normalize_text(ground_truth)
            pred = self.normalize_text(prediction)
        else:
            gt = ground_truth
            pred = prediction
        
        # Remove spaces for character comparison
        gt_chars = gt.replace(" ", "")
        pred_chars = pred.replace(" ", "")
        
        # Calculate character edit distance
        matcher = SequenceMatcher(None, gt_chars, pred_chars)
        ratio = matcher.ratio()
        
        # Count exact matches
        matches = sum(1 for a, b in zip(gt_chars, pred_chars) if a == b)
        max_len = max(len(gt_chars), len(pred_chars), 1)
        
        return {
            "similarity_ratio": ratio,
            "character_accuracy": matches / max_len if max_len > 0 else 0,
            "ground_truth_length": len(gt_chars),
            "prediction_length": len(pred_chars),
            "length_difference": abs(len(gt_chars) - len(pred_chars)),
        }
    
    def word_accuracy(
        self,
        ground_truth: str,
        prediction: str,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Calculate word-level accuracy metrics.
        
        Args:
            ground_truth: Expected transliteration
            prediction: VLM-generated transliteration
            normalize: Whether to normalize texts before comparison
            
        Returns:
            Dictionary with word-level metrics
        """
        if normalize:
            gt = self.normalize_text(ground_truth)
            pred = self.normalize_text(prediction)
        else:
            gt = ground_truth
            pred = prediction
        
        gt_words = gt.split()
        pred_words = pred.split()
        
        # Calculate word matches
        matches = sum(1 for a, b in zip(gt_words, pred_words) if a == b)
        max_words = max(len(gt_words), len(pred_words), 1)
        
        # Use sequence matcher on word sequence
        matcher = SequenceMatcher(None, gt_words, pred_words)
        word_ratio = matcher.ratio()
        
        return {
            "word_accuracy": matches / max_words if max_words > 0 else 0,
            "word_sequence_ratio": word_ratio,
            "ground_truth_words": len(gt_words),
            "prediction_words": len(pred_words),
            "matching_words": matches,
        }
    
    def semantic_similarity(
        self,
        ground_truth_translation: str,
        prediction_translation: str
    ) -> Dict[str, any]:
        """
        Assess semantic similarity of translations.
        
        Args:
            ground_truth_translation: Expected translation
            prediction_translation: VLM-generated translation
            
        Returns:
            Dictionary with semantic metrics
        """
        gt = self.normalize_text(ground_truth_translation)
        pred = self.normalize_text(prediction_translation)
        
        # Basic keyword extraction
        def extract_keywords(text: str) -> set:
            """Extract meaningful keywords from translation."""
            # Common Safaitic content words
            important_words = {
                'son', 'daughter', 'father', 'mother', 'brother', 'sister',
                'by', 'camped', 'herded', 'raided', 'grieved', 'buried',
                'security', 'peace', 'booty', 'friend', 'year'
            }
            words = set(text.split())
            return words & important_words
        
        gt_keywords = extract_keywords(gt)
        pred_keywords = extract_keywords(pred)
        
        # Keyword overlap
        if gt_keywords or pred_keywords:
            keyword_precision = len(gt_keywords & pred_keywords) / len(pred_keywords) if pred_keywords else 0
            keyword_recall = len(gt_keywords & pred_keywords) / len(gt_keywords) if gt_keywords else 0
            keyword_f1 = 2 * (keyword_precision * keyword_recall) / (keyword_precision + keyword_recall) \
                if (keyword_precision + keyword_recall) > 0 else 0
        else:
            keyword_precision = keyword_recall = keyword_f1 = 0
        
        # Overall text similarity
        text_ratio = SequenceMatcher(None, gt, pred).ratio()
        
        return {
            "text_similarity": text_ratio,
            "keyword_precision": keyword_precision,
            "keyword_recall": keyword_recall,
            "keyword_f1": keyword_f1,
            "ground_truth_keywords": list(gt_keywords),
            "prediction_keywords": list(pred_keywords),
            "matching_keywords": list(gt_keywords & pred_keywords),
        }
    
    def evaluate_inscription(
        self,
        ground_truth_transliteration: str,
        ground_truth_translation: str,
        vlm_response: str,
        extract_transliteration: bool = True
    ) -> Dict[str, any]:
        """
        Comprehensive evaluation of VLM response.
        
        Args:
            ground_truth_transliteration: Expected transliteration
            ground_truth_translation: Expected translation
            vlm_response: Complete VLM response text
            extract_transliteration: Whether to extract transliteration from response
            
        Returns:
            Comprehensive evaluation metrics
        """
        # Try to extract transliteration from response
        if extract_transliteration:
            pred_translit = self.extract_transliteration(vlm_response)
        else:
            pred_translit = vlm_response
        
        # Character-level metrics
        char_metrics = self.character_accuracy(
            ground_truth_transliteration,
            pred_translit
        )
        
        # Word-level metrics
        word_metrics = self.word_accuracy(
            ground_truth_transliteration,
            pred_translit
        )
        
        # Semantic metrics (if translation provided in VLM response)
        semantic_metrics = self.semantic_similarity(
            ground_truth_translation,
            vlm_response
        )
        
        # Overall assessment
        overall_score = (
            char_metrics['similarity_ratio'] * 0.4 +
            word_metrics['word_accuracy'] * 0.4 +
            semantic_metrics['keyword_f1'] * 0.2
        )
        
        return {
            "overall_score": overall_score,
            "character_metrics": char_metrics,
            "word_metrics": word_metrics,
            "semantic_metrics": semantic_metrics,
            "ground_truth": {
                "transliteration": ground_truth_transliteration,
                "translation": ground_truth_translation,
            },
            "prediction": {
                "extracted_transliteration": pred_translit,
                "full_response": vlm_response,
            }
        }
    
    def check_script_identification(self, vlm_response: str) -> Dict[str, any]:
        """
        Check if VLM correctly identified the script as Safaitic.
        
        Args:
            vlm_response: VLM response text
            
        Returns:
            Dictionary with identification results
        """
        response_lower = vlm_response.lower()
        
        # Look for mentions of Safaitic or related terms
        safaitic_terms = ['safaitic', 'safaïtic', 'ancient north arabian', 'north arabian']
        arabic_terms = ['arabic', 'arab']
        semitic_terms = ['semitic']
        
        safaitic_mentioned = any(term in response_lower for term in safaitic_terms)
        arabic_mentioned = any(term in response_lower for term in arabic_terms)
        semitic_mentioned = any(term in response_lower for term in semitic_terms)
        
        # Check for script characteristics
        mentions_rtl = 'right-to-left' in response_lower or 'right to left' in response_lower
        mentions_inscr = 'inscription' in response_lower or 'carved' in response_lower
        
        confidence = 0
        if safaitic_mentioned:
            confidence = 1.0
        elif arabic_mentioned and (mentions_rtl or mentions_inscr):
            confidence = 0.7  # Partially correct
        elif semitic_mentioned:
            confidence = 0.5
        elif mentions_rtl or mentions_inscr:
            confidence = 0.3
        
        return {
            "correctly_identified": safaitic_mentioned,
            "confidence": confidence,
            "safaitic_mentioned": safaitic_mentioned,
            "arabic_mentioned": arabic_mentioned,
            "semitic_mentioned": semitic_mentioned,
            "mentions_rtl": mentions_rtl,
            "mentions_inscription": mentions_inscr,
        }
    
    def batch_evaluate(
        self,
        evaluations: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Aggregate results from multiple evaluations.
        
        Args:
            evaluations: List of evaluation dictionaries
            
        Returns:
            Aggregated statistics
        """
        if not evaluations:
            return {}
        
        # Calculate averages
        avg_overall = sum(e['overall_score'] for e in evaluations) / len(evaluations)
        avg_char_sim = sum(e['character_metrics']['similarity_ratio'] for e in evaluations) / len(evaluations)
        avg_word_acc = sum(e['word_metrics']['word_accuracy'] for e in evaluations) / len(evaluations)
        avg_semantic = sum(e['semantic_metrics']['keyword_f1'] for e in evaluations) / len(evaluations)
        
        return {
            "num_evaluations": len(evaluations),
            "average_overall_score": avg_overall,
            "average_character_similarity": avg_char_sim,
            "average_word_accuracy": avg_word_acc,
            "average_semantic_f1": avg_semantic,
            "score_distribution": {
                "excellent (>0.8)": sum(1 for e in evaluations if e['overall_score'] > 0.8),
                "good (0.6-0.8)": sum(1 for e in evaluations if 0.6 <= e['overall_score'] <= 0.8),
                "fair (0.4-0.6)": sum(1 for e in evaluations if 0.4 <= e['overall_score'] < 0.6),
                "poor (<0.4)": sum(1 for e in evaluations if e['overall_score'] < 0.4),
            },
            "individual_results": evaluations,
        }
