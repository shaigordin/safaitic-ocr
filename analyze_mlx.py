#!/usr/bin/env python3
"""
Safaitic OCR Analysis using MLX-VLM
Uses mlx-vlm library for efficient local VLM inference on Apple Silicon
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from PIL import Image

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from src.utils import load_metadata, get_inscription_data
from src.prompt_templates import SafaiticPrompts


def resize_image_for_vlm(image_path: Path, max_size: int = 1024) -> Path:
    """
    Resize image if too large to avoid memory issues.
    
    Args:
        image_path: Path to original image
        max_size: Maximum dimension (width or height)
    
    Returns:
        Path to resized image (or original if already small enough)
    """
    img = Image.open(image_path)
    
    # Check if resizing is needed
    if max(img.size) <= max_size:
        return image_path
    
    # Calculate new dimensions
    ratio = max_size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    
    # Resize
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Save to temp location
    temp_path = Path("/tmp") / f"resized_{image_path.name}"
    img_resized.save(temp_path, quality=95)
    
    return temp_path


def analyze_inscription_mlx(
    model,
    processor,
    config,
    inscription,
    prompts: List[Dict],
    verbose: bool = False
) -> Dict:
    """
    Analyze a single inscription with MLX-VLM
    
    Args:
        model: MLX-VLM model
        processor: MLX-VLM processor
        config: Model configuration
        inscription: Inscription data with images
        prompts: List of prompt dictionaries
        verbose: Whether to print progress
    
    Returns:
        Dictionary with analysis results
    """
    results = {
        "inscription_siglum": inscription.inscription_id,
        "num_images": len(inscription.image_files),
        "ground_truth": {
            "transliteration": inscription.transliteration,
            "translation": inscription.translation
        },
        "prompts": []
    }
    
    # Use first image for analysis (can be extended to multi-image)
    # Build image path from inscription data
    if not inscription.image_files:
        image_path = None
    else:
        folder_name = inscription.inscription_id
        image_filename = inscription.image_files[0]
        image_path = Path("data") / "examples" / "BES15" / folder_name / image_filename
        # Resize image to avoid memory issues
        image_path = resize_image_for_vlm(image_path, max_size=1024)
    
    for prompt_info in prompts:
        prompt_name = prompt_info["name"]
        prompt_text = prompt_info["prompt"]
        
        if verbose:
            print(f"\n  Prompt: {prompt_name}")
            print(f"    {prompt_text[:100]}...")
        
        try:
            # Apply chat template with image
            formatted_prompt = apply_chat_template(
                processor, 
                config, 
                prompt_text,
                num_images=1 if image_path else 0
            )
            
            # Generate response
            output = generate(
                model, 
                processor, 
                formatted_prompt,
                image=[str(image_path)] if image_path else None,
                max_tokens=500,
                temperature=0.1,
                verbose=False
            )
            
            # Handle output - may be string or object with text attribute
            if isinstance(output, str):
                response_text = output
            else:
                response_text = getattr(output, 'text', str(output))
            
            # Ensure UTF-8 compatibility
            response_text = response_text.encode('utf-8', errors='replace').decode('utf-8')
            
            result = {
                "prompt_name": prompt_name,
                "prompt": prompt_text,
                "response": response_text,
                "duration_seconds": getattr(output, 'generation_time', 0),
                "success": True,
                "error": None
            }
            
            if verbose:
                print(f"    ✓ Success ({result['duration_seconds']:.1f}s)")
                print(f"    Response: {response_text[:150]}...")
                
        except Exception as e:
            result = {
                "prompt_name": prompt_name,
                "prompt": prompt_text,
                "response": None,
                "duration_seconds": 0,
                "success": False,
                "error": str(e)
            }
            
            if verbose:
                print(f"    ✗ Failed: {str(e)}")
        
        results["prompts"].append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Safaitic inscriptions using MLX-VLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        help="MLX model path (default: Qwen2.5-VL-7B-Instruct-4bit)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="metadata/BES15.csv",
        help="Path to metadata CSV"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing inscription images"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of inscriptions to process (default: 10)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all inscriptions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docs/data",
        help="Directory to save results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n{'='*70}")
    print(f"LOADING MLX-VLM MODEL")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    
    model, processor = load(args.model)
    config = load_config(args.model)
    
    print("✓ Model loaded successfully")
    
    # Load inscription data
    print(f"\n{'='*70}")
    print(f"LOADING INSCRIPTION DATA")
    print(f"{'='*70}")
    
    df = load_metadata(args.metadata)
    
    # Get unique inscriptions with images
    inscriptions_with_images = []
    data_path = Path(args.data_dir)
    
    for siglum in df['inscription_siglum'].unique():
        try:
            inscription = get_inscription_data(df, args.data_dir, siglum, load_images=False)
            if inscription.image_files:  # Check image_files instead of images
                inscriptions_with_images.append(inscription)
        except:
            continue
    
    print(f"✓ Found {len(inscriptions_with_images)} inscriptions with images")
    
    # Determine how many to process
    if args.all:
        count = len(inscriptions_with_images)
    else:
        count = min(args.count, len(inscriptions_with_images))
    
    print(f"  Processing first {count}")
    
    # Prepare prompts
    prompts = [
        {
            "name": "description",
            "prompt": SafaiticPrompts.basic_description()
        },
        {
            "name": "script_id",
            "prompt": SafaiticPrompts.script_identification()
        },
        {
            "name": "transliteration",
            "prompt": SafaiticPrompts.transliteration_attempt()
        }
    ]
    
    # Process inscriptions
    print(f"\n{'='*70}")
    print(f"ANALYZING INSCRIPTIONS")
    print(f"{'='*70}")
    
    all_results = []
    successful = 0
    
    for i, inscription in enumerate(inscriptions_with_images[:count], 1):
        print(f"\n[{i}/{count}]")
        print(f"{'='*70}")
        print(f"Inscription: {inscription.inscription_id}")
        print(f"Images: {len(inscription.image_files)}")
        print(f"{'='*70}")
        
        result = analyze_inscription_mlx(
            model,
            processor,
            config,
            inscription,
            prompts,
            verbose=args.verbose
        )
        
        all_results.append(result)
        
        # Count successful prompts
        successful_prompts = sum(1 for p in result["prompts"] if p["success"])
        if successful_prompts == len(prompts):
            successful += 1
    
    # Save results
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detailed results
    detailed_file = output_dir / f"mlx_results_{timestamp}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "model": args.model,
                "backend": "mlx-vlm",
                "num_inscriptions": count,
                "num_prompts": len(prompts)
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved detailed results: {detailed_file}")
    
    # Latest results (for web app)
    latest_file = output_dir / "latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "model": args.model,
                "backend": "mlx-vlm",
                "num_inscriptions": count,
                "num_prompts": len(prompts)
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved latest results: {latest_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total processed: {count}")
    print(f"Successful: {successful}")
    print(f"Failed: {count - successful}")
    if count > 0:
        print(f"Success rate: {successful/count*100:.1f}%")
    else:
        print(f"Success rate: N/A (no inscriptions processed)")
    
    print(f"\nNext steps:")
    print(f"  1. Review results in {latest_file}")
    print(f"  2. Open docs/index.html to view web application")
    print(f"  3. Deploy to GitHub Pages (git push)")


if __name__ == "__main__":
    main()
