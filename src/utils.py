"""
Utility functions for loading and processing Safaitic inscription data.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
from PIL import Image


@dataclass
class InscriptionData:
    """Container for Safaitic inscription data."""
    
    inscription_id: str
    siglum: str
    transliteration: str
    translation: str
    provenance: str
    ociana_link: str
    image_files: List[str]
    images: Optional[List[Image.Image]] = None
    
    def __repr__(self):
        return f"InscriptionData(id={self.inscription_id}, images={len(self.image_files)})"


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load inscription metadata from CSV file.
    
    Args:
        csv_path: Path to the metadata CSV file
        
    Returns:
        DataFrame with inscription metadata
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")
    print(f"Unique inscriptions: {df['inscription_siglum'].nunique()}")
    return df


def get_inscription_by_siglum(df: pd.DataFrame, siglum: str) -> pd.DataFrame:
    """
    Get all records for a specific inscription siglum.
    
    Args:
        df: Metadata DataFrame
        siglum: Inscription siglum (e.g., "BES15 1")
        
    Returns:
        DataFrame with all records for that inscription
    """
    return df[df['inscription_siglum'] == siglum]


def load_inscription_images(
    data_dir: str,
    inscription_folder: str,
    image_filenames: Optional[List[str]] = None
) -> List[Image.Image]:
    """
    Load images for a specific inscription.
    
    Args:
        data_dir: Base data directory path
        inscription_folder: Folder name for the inscription
        image_filenames: Optional list of specific filenames to load
        
    Returns:
        List of PIL Image objects
    """
    # BES15 inscriptions are in data/examples/BES15/{inscription}
    folder_path = Path(data_dir) / "examples" / "BES15" / inscription_folder
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Inscription folder not found: {folder_path}")
    
    if image_filenames:
        # Load specific files
        image_files = [folder_path / fname for fname in image_filenames]
    else:
        # Load all jpg files (exclude thumbnails and cards)
        image_files = [
            f for f in folder_path.glob("*.jpg")
            if not f.name.startswith(("thumb_", "card_", "_"))
        ]
    
    images = []
    for img_path in sorted(image_files):
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
    
    print(f"Loaded {len(images)} images from {inscription_folder}")
    return images


def get_inscription_data(
    df: pd.DataFrame,
    data_dir: str,
    siglum: str,
    load_images: bool = True
) -> InscriptionData:
    """
    Get complete inscription data including metadata and images.
    
    Args:
        df: Metadata DataFrame
        data_dir: Base data directory path
        siglum: Inscription siglum (e.g., "BES15 1")
        load_images: Whether to load the actual image data
        
    Returns:
        InscriptionData object
    """
    records = get_inscription_by_siglum(df, siglum)
    
    if records.empty:
        raise ValueError(f"No inscription found with siglum: {siglum}")
    
    # Get unique values (should be same across all records for this inscription)
    first_record = records.iloc[0]
    image_filenames = records['filename'].tolist()
    
    # Determine folder name from siglum
    folder_name = siglum.replace(" ", " ")
    
    images = None
    if load_images:
        try:
            images = load_inscription_images(data_dir, folder_name, image_filenames)
        except FileNotFoundError:
            print(f"Warning: Images not found for {siglum}")
    
    return InscriptionData(
        inscription_id=siglum,
        siglum=first_record['reference_siglum'] if pd.notna(first_record['reference_siglum']) else siglum,
        transliteration=first_record['transliteration'] if pd.notna(first_record['transliteration']) else "",
        translation=first_record['translation'] if pd.notna(first_record['translation']) else "",
        provenance=first_record['site'] if pd.notna(first_record['site']) else "",
        ociana_link=first_record['ociana_link'] if pd.notna(first_record['ociana_link']) else "",
        image_files=image_filenames,
        images=images
    )


def prepare_image_for_vlm(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """
    Prepare an image for VLM processing by resizing if needed.
    
    Args:
        image: PIL Image object
        max_size: Maximum dimension (width or height)
        
    Returns:
        Processed PIL Image
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize if too large
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    return image


def list_available_inscriptions(df: pd.DataFrame) -> List[str]:
    """
    Get list of all available inscription sigla.
    
    Args:
        df: Metadata DataFrame
        
    Returns:
        Sorted list of unique inscription sigla
    """
    return sorted(df['inscription_siglum'].unique().tolist())
