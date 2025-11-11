"""
Safaitic OCR - VLM Testing Pipeline
====================================

A toolkit for testing Vision Language Models (VLMs) on ancient Safaitic inscriptions.
This package provides interfaces to VLMs, evaluation metrics, and utilities for
working with Safaitic script images and metadata.
"""

__version__ = "0.1.0"
__author__ = "Safaitic OCR Project"

from .utils import (
    load_metadata,
    load_inscription_images,
    get_inscription_data,
    list_available_inscriptions,
    prepare_image_for_vlm,
    InscriptionData
)
from .vlm_interface import LlamaVision
from .gradio_vlm_interface import GradioSpaceVLM
from .prompt_templates import SafaiticPrompts
from .evaluator import InscriptionEvaluator

__all__ = [
    "load_metadata",
    "load_inscription_images",
    "get_inscription_data",
    "list_available_inscriptions",
    "prepare_image_for_vlm",
    "InscriptionData",
    "LlamaVision",
    "GradioSpaceVLM",
    "SafaiticPrompts",
    "InscriptionEvaluator",
]
