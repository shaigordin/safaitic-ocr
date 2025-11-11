#!/usr/bin/env python3
"""
Generate VLM analysis results for Safaitic inscriptions using FREE local Ollama.

This script analyzes inscriptions using Ollama (completely FREE, runs locally)
and generates results for the web application.

Usage:
    python generate_results.py              # Test mode: 10 inscriptions
    python generate_results.py --all        # Full dataset: all inscriptions
    python generate_results.py --count 50   # Specific number
    python generate_results.py --use-gradio # Use Gradio Spaces instead of Ollama
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src import (
    load_metadata,
    get_inscription_data,
    LlamaVision,
    GradioSpaceVLM,
    SafaiticPrompts
)


# Configuration
METADATA_FILE = "metadata/BES15.csv"
DATA_DIR = "data"
OUTPUT_DIR = "docs/data"
TEST_MODE_COUNT = 10  # Number of inscriptions in test mode

# Prompts to use for each inscription
PROMPTS = {
    "description": SafaiticPrompts.basic_description(),
    "script_id": SafaiticPrompts.script_identification(),
    "transliteration": SafaiticPrompts.transliteration_attempt(),
}


def build_ociana_url(siglum: str, filename: str) -> str:
    """Build OCIANA image URL."""
    # Format: https://ociana.osu.edu/uploads/inscription_image/image/BES15%20123/filename.jpg
    siglum_encoded = siglum.replace(" ", "%20")
    return f"https://ociana.osu.edu/uploads/inscription_image/image/{siglum_encoded}/{filename}"


def analyze_inscription(
    vlm,  # Can be LlamaVision or GradioSpaceVLM
    inscription_data,
    prompts: Dict[str, str]
) -> Dict:
    """
    Analyze a single inscription with multiple prompts.
    
    Returns:
        Dictionary with inscription data and VLM analysis results
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {inscription_data.siglum}")
    print(f"Images: {len(inscription_data.images)}")
    print('='*70)
    
    results = {
        "siglum": inscription_data.siglum,
        "transliteration_ground_truth": inscription_data.transliteration,
        "translation_ground_truth": inscription_data.translation,
        "num_images": len(inscription_data.images),
        "image_files": inscription_data.image_files,
        "ociana_urls": [
            build_ociana_url(inscription_data.siglum, img_file)
            for img_file in inscription_data.image_files
        ],
        "analyses": {}
    }
    
    # Analyze with each prompt
    for prompt_name, prompt_text in prompts.items():
        print(f"\nPrompt: {prompt_name}")
        print(f"  {prompt_text[:80]}...")
        
        # Use first image (or could analyze all and combine)
        image = inscription_data.images[0]
        result = vlm.analyze_image(image, prompt_text)
        
        # Convert Ollama's nanosecond duration to seconds
        duration_ns = result.get('total_duration', 0)
        duration_s = duration_ns / 1_000_000_000 if duration_ns else None
        
        if result['success']:
            print(f"  ✓ Success ({duration_s:.1f}s)")
            print(f"  Response: {result['response'][:100]}...")
        else:
            print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        results["analyses"][prompt_name] = {
            "prompt": prompt_text,
            "response": result.get('response', ''),
            "success": result['success'],
            "error": result.get('error'),
            "duration_seconds": duration_s,
            "model": result.get('model')
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate VLM analysis results for Safaitic inscriptions"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all inscriptions (default: test mode with 10)"
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Process specific number of inscriptions"
    )
    parser.add_argument(
        "--use-gradio",
        action="store_true",
        help="Use Gradio Spaces instead of local Ollama (experimental)"
    )
    parser.add_argument(
        "--space",
        type=str,
        default="llava-onevision",
        help="Gradio Space to use when --use-gradio is specified (default: llava-onevision)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-vision",
        help="Ollama model name (default: llama3.2-vision)"
    )
    
    args = parser.parse_args()
    
    # Determine how many inscriptions to process
    if args.count:
        num_to_process = args.count
        mode = f"{num_to_process} inscriptions"
    elif args.all:
        num_to_process = None
        mode = "ALL inscriptions"
    else:
        num_to_process = TEST_MODE_COUNT
        mode = f"TEST MODE ({TEST_MODE_COUNT} inscriptions)"
    
    print("=" * 70)
    print("SAFAITIC VLM ANALYSIS - FREE LOCAL OLLAMA")
    print("=" * 70)
    print(f"Mode: {mode}")
    if args.use_gradio:
        print(f"VLM Backend: Gradio Space ({args.space})")
    else:
        print(f"VLM Backend: Local Ollama ({args.model})")
    print(f"Metadata: {METADATA_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Load metadata
    print("Loading metadata...")
    df = load_metadata(METADATA_FILE)
    total_inscriptions = len(df)
    print(f"✓ Loaded {total_inscriptions} records")
    
    # Get unique inscriptions with images
    unique_sigla = df['inscription_siglum'].unique()
    print(f"✓ Found {len(unique_sigla)} unique inscriptions")
    
    inscriptions_to_process = []
    for siglum in unique_sigla:
        try:
            inscription = get_inscription_data(df, DATA_DIR, siglum, load_images=True)
            if inscription.images:
                inscriptions_to_process.append(siglum)
        except Exception as e:
            pass  # Silently skip inscriptions without images
    
    print(f"✓ Found {len(inscriptions_to_process)} inscriptions with images")
    
    # Limit to requested count
    if num_to_process:
        inscriptions_to_process = inscriptions_to_process[:num_to_process]
        print(f"  Processing first {len(inscriptions_to_process)}")
    
    # Initialize VLM
    print(f"\nInitializing VLM...")
    try:
        if args.use_gradio:
            print(f"  Using Gradio Space: {args.space}")
            vlm = GradioSpaceVLM(space_id=args.space)
        else:
            print(f"  Using local Ollama: {args.model}")
            vlm = LlamaVision(model_name=args.model)
        print()
    except Exception as e:
        print(f"✗ Failed to initialize VLM: {e}")
        return 1
    
    # Check availability
    if not vlm.check_availability():
        print("\n⚠️  VLM may not be available.")
        if not args.use_gradio:
            print("\nMake sure Ollama is running:")
            print("  1. Install: brew install ollama")
            print("  2. Pull model: ollama pull llama3.2-vision")
            print("  3. Check: ollama list")
        return 1
    print()
    
    # Process inscriptions
    results_list = []
    successful = 0
    failed = 0
    
    for i, siglum in enumerate(inscriptions_to_process, 1):
        print(f"\n[{i}/{len(inscriptions_to_process)}]")
        
        try:
            inscription = get_inscription_data(df, DATA_DIR, siglum, load_images=True)
            result = analyze_inscription(vlm, inscription, PROMPTS)
            results_list.append(result)
            successful += 1
        except Exception as e:
            print(f"✗ Error processing {siglum}: {e}")
            failed += 1
            continue
    
    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print('='*70)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    output_file = output_dir / f"results_{timestamp}.json"
    
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "mode": mode,
            "backend": "gradio" if args.use_gradio else "ollama",
            "space": args.space if args.use_gradio else None,
            "model": args.model if not args.use_gradio else vlm.model_name,
            "total_processed": len(inscriptions_to_process),
            "successful": successful,
            "failed": failed,
            "prompts_used": list(PROMPTS.keys())
        },
        "results": results_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved detailed results: {output_file}")
    
    # Save latest.json (for web app)
    latest_file = output_dir / "latest.json"
    with open(latest_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved latest results: {latest_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    print(f"Total processed: {len(inscriptions_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(inscriptions_to_process)*100:.1f}%")
    print()
    print("Next steps:")
    print("  1. Review results in docs/data/latest.json")
    print("  2. Open docs/index.html to view web application")
    print("  3. Deploy to GitHub Pages (git push)")
    print()
    
    return 0


if __name__ == "__main__":
    exit(main())
