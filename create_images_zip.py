#!/usr/bin/env python3
"""
Create a zip file of inscription images for GitHub deployment.

This script creates a zip file containing all images from the 50 inscriptions
used in the evaluation, organized by inscription siglum. The zip can be hosted
on GitHub releases or extracted to docs/images/ for the web demo.

Usage:
    python create_images_zip.py

Output:
    Creates docs/inscription_images.zip with structure:
    BES15 1/
        im*.jpg
    BES15 2/
        im*.jpg
    ...
"""

import json
import zipfile
from pathlib import Path


def get_evaluated_inscriptions(json_file: Path) -> set:
    """Extract list of inscription siglums from evaluation JSON."""
    with open(json_file) as f:
        data = json.load(f)
    return {result['inscription_siglum'] for result in data['results']}


def create_images_zip(
    source_dir: Path,
    output_zip: Path,
    inscriptions: set,
    image_pattern: str = 'im*.jpg'
):
    """
    Create zip file containing images for specified inscriptions.
    
    Args:
        source_dir: Path to data/examples/BES15/ directory
        output_zip: Path to output zip file
        inscriptions: Set of inscription siglums to include
        image_pattern: Glob pattern for image files (default: im*.jpg for full-size images)
    """
    print(f"Creating zip archive: {output_zip}")
    print(f"Including {len(inscriptions)} inscriptions")
    
    total_files = 0
    total_size = 0
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for siglum in sorted(inscriptions):
            inscription_dir = source_dir / siglum
            
            if not inscription_dir.exists():
                print(f"Warning: Directory not found: {inscription_dir}")
                continue
            
            # Get all matching images
            images = list(inscription_dir.glob(image_pattern))
            
            if not images:
                print(f"Warning: No images found in {inscription_dir}")
                continue
            
            # Add each image to zip, preserving directory structure
            for img_path in images:
                arcname = f"{siglum}/{img_path.name}"
                zf.write(img_path, arcname)
                total_files += 1
                total_size += img_path.stat().st_size
            
            print(f"  {siglum}: {len(images)} images")
    
    # Print summary
    zip_size = output_zip.stat().st_size
    compression_ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print(f"\nZip creation complete!")
    print(f"  Total files: {total_files}")
    print(f"  Original size: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Zip size: {zip_size / 1024 / 1024:.1f} MB")
    print(f"  Compression: {compression_ratio:.1f}%")
    print(f"\nTo use in website:")
    print(f"  1. Upload {output_zip.name} to GitHub releases")
    print(f"  2. Extract to docs/images/ directory")
    print(f"  3. Update index.html image paths to: ../images/{{siglum}}/im*.jpg")


def main():
    # Paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    examples_dir = data_dir / "examples" / "BES15"
    docs_dir = project_root / "docs"
    
    # Get inscriptions from any of the evaluation JSON files
    json_file = docs_dir / "data" / "qwen2-2b_50inscriptions.json"
    
    if not json_file.exists():
        print(f"Error: Evaluation JSON not found: {json_file}")
        return 1
    
    if not examples_dir.exists():
        print(f"Error: Images directory not found: {examples_dir}")
        return 1
    
    # Get list of evaluated inscriptions
    inscriptions = get_evaluated_inscriptions(json_file)
    print(f"Found {len(inscriptions)} inscriptions in evaluation")
    
    # Create zip file
    output_zip = docs_dir / "inscription_images.zip"
    create_images_zip(examples_dir, output_zip, inscriptions)
    
    return 0


if __name__ == "__main__":
    exit(main())
