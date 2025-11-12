#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "huggingface-hub[hf_transfer]",
#     "torch",
#     "transformers",
#     "vllm",
#     "Pillow"
# ]
# ///
"""
Safaitic OCR Batch Analysis using HuggingFace Jobs with UV Scripts

This script follows the uv-scripts pattern for serverless OCR processing
on HuggingFace Jobs infrastructure. Based on:
- https://huggingface.co/datasets/uv-scripts/ocr

Usage:
    hf jobs uv run --flavor a100-large \\
        uv_batch_analysis.py \\
        shaigordin/safaitic-inscriptions \\
        shaigordin/safaitic-ocr-results \\
        --max-samples 1401
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset, Dataset as HFDataset
from huggingface_hub import HfApi
from PIL import Image
from vllm import LLM, SamplingParams


def get_safaitic_prompts() -> List[Dict[str, str]]:
    """
    Get Safaitic-specific prompts for VLM analysis
    
    Returns:
        List of prompt dictionaries with name and text
    """
    return [
        {
            "name": "description",
            "prompt": """Describe what you see in this image in detail. 
Focus on any text, inscriptions, or writing visible on the surface."""
        },
        {
            "name": "script_identification",
            "prompt": """Analyze this image carefully.

1. Is there any text or inscription visible?
2. If yes, what writing system or script does it appear to be? 
3. Describe the characteristics of the writing (direction, shape of letters, etc.)
4. What is the surface material and condition?"""
        },
        {
            "name": "transliteration",
            "prompt": """Safaitic letters: ʾ b g d h w z ḥ ṭ y k l m n s ʿ f ṣ q r t ṯ ḫ ḏ ḍ ġ. 
Often starts with: l (for/to), w (and), h- (this/the).

Analyze this Safaitic inscription:
1. Identify visible characters using the Safaitic alphabet
2. Provide transliteration in Latin letters
3. Note reading direction
4. Indicate any unclear or damaged sections with [?]"""
        }
    ]


def analyze_image_with_vlm(
    llm: LLM,
    image: Image.Image,
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.1
) -> Dict:
    """
    Analyze a single image with VLM
    
    Args:
        llm: vLLM inference engine
        image: PIL Image object
        prompt: Text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Dictionary with response and metadata
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9
    )
    
    try:
        # Prepare input for vLLM
        # Note: Format depends on model - this is generic
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            }
        }
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        
        if outputs and len(outputs) > 0:
            output = outputs[0]
            return {
                "response": output.outputs[0].text,
                "success": True,
                "error": None,
                "tokens_generated": len(output.outputs[0].token_ids)
            }
        else:
            return {
                "response": None,
                "success": False,
                "error": "No output generated",
                "tokens_generated": 0
            }
            
    except Exception as e:
        return {
            "response": None,
            "success": False,
            "error": str(e),
            "tokens_generated": 0
        }


def process_dataset(
    input_dataset: str,
    output_dataset: str,
    model_name: str = "deepseek-ai/DeepSeek-OCR",
    image_column: str = "image",
    max_samples: Optional[int] = None,
    batch_size: int = 16,
    max_tokens: int = 500,
    temperature: float = 0.1
):
    """
    Process Safaitic inscription dataset with VLM
    
    Args:
        input_dataset: HuggingFace dataset with images
        output_dataset: Output dataset name
        model_name: VLM model to use
        image_column: Column name containing images
        max_samples: Limit number of samples (for testing)
        batch_size: Batch size for inference
        max_tokens: Max tokens per generation
        temperature: Sampling temperature
    """
    print(f"\n{'='*70}")
    print(f"SAFAITIC OCR BATCH ANALYSIS")
    print(f"{'='*70}")
    print(f"Input dataset: {input_dataset}")
    print(f"Output dataset: {output_dataset}")
    print(f"Model: {model_name}")
    print(f"Max samples: {max_samples if max_samples else 'all'}")
    
    # Load input dataset
    print(f"\nLoading dataset...")
    dataset = load_dataset(input_dataset, split="train")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Load VLM model
    print(f"\nLoading VLM model: {model_name}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.9
    )
    print(f"✓ Model loaded")
    
    # Get prompts
    prompts = get_safaitic_prompts()
    print(f"\n✓ Prepared {len(prompts)} analysis prompts")
    
    # Process each sample
    print(f"\n{'='*70}")
    print(f"PROCESSING INSCRIPTIONS")
    print(f"{'='*70}")
    
    results = []
    
    for idx, sample in enumerate(dataset):
        inscription_id = sample.get("inscription_siglum", f"inscription_{idx}")
        image = sample[image_column]
        
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        print(f"\n[{idx+1}/{len(dataset)}] Processing: {inscription_id}")
        
        sample_results = {
            "inscription_siglum": inscription_id,
            "ground_truth_transliteration": sample.get("transliteration", None),
            "ground_truth_translation": sample.get("translation", None),
            "prompts": []
        }
        
        # Analyze with each prompt
        for prompt_info in prompts:
            print(f"  - {prompt_info['name']}...", end=" ")
            
            result = analyze_image_with_vlm(
                llm,
                image,
                prompt_info["prompt"],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            result.update({
                "prompt_name": prompt_info["name"],
                "prompt": prompt_info["prompt"]
            })
            
            sample_results["prompts"].append(result)
            
            if result["success"]:
                print("✓")
            else:
                print(f"✗ {result['error']}")
        
        results.append(sample_results)
    
    # Create output dataset
    print(f"\n{'='*70}")
    print(f"SAVING RESULTS")
    print(f"{'='*70}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_data = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_name,
            "backend": "vllm-hf-jobs",
            "num_inscriptions": len(results),
            "num_prompts": len(prompts)
        },
        "results": results
    }
    
    # Convert to HuggingFace dataset format
    output_records = []
    for result in results:
        output_records.append({
            "inscription_siglum": result["inscription_siglum"],
            "ground_truth_transliteration": result.get("ground_truth_transliteration"),
            "ground_truth_translation": result.get("ground_truth_translation"),
            "analysis_results": json.dumps(result["prompts"], ensure_ascii=False)
        })
    
    output_hf_dataset = HFDataset.from_dict({
        key: [record[key] for record in output_records]
        for key in output_records[0].keys()
    })
    
    # Push to Hub
    print(f"Pushing to HuggingFace Hub: {output_dataset}")
    output_hf_dataset.push_to_hub(
        output_dataset,
        private=False,
        commit_message=f"Safaitic OCR analysis with {model_name} - {timestamp}"
    )
    
    print(f"✓ Results saved to: https://huggingface.co/datasets/{output_dataset}")
    
    # Summary
    total_prompts = len(results) * len(prompts)
    successful_prompts = sum(
        sum(1 for p in r["prompts"] if p["success"])
        for r in results
    )
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total inscriptions: {len(results)}")
    print(f"Total prompts processed: {total_prompts}")
    print(f"Successful: {successful_prompts}")
    print(f"Failed: {total_prompts - successful_prompts}")
    print(f"Success rate: {successful_prompts/total_prompts*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Safaitic OCR batch analysis with VLM on HF Jobs"
    )
    parser.add_argument(
        "input_dataset",
        type=str,
        help="Input HuggingFace dataset with inscriptions"
    )
    parser.add_argument(
        "output_dataset",
        type=str,
        help="Output HuggingFace dataset name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-OCR",
        help="VLM model to use (default: DeepSeek-OCR)"
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="Column containing images"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        model_name=args.model,
        image_column=args.image_column,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()
