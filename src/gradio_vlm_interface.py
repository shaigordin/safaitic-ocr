"""
Interface for Vision Language Models via HuggingFace Gradio Spaces.

This module provides FREE VLM inference using HuggingFace Spaces with
an API compatible with the existing LlamaVision interface.
"""

import tempfile
import time
from pathlib import Path
from typing import Union, List, Dict, Optional
from PIL import Image
import io


class GradioSpaceVLM:
    """
    Interface for Vision Language Models via HuggingFace Gradio Spaces.
    
    Provides FREE inference using public Gradio Spaces. Compatible API
    with LlamaVision for easy integration.
    """
    
    AVAILABLE_SPACES = {
        "llava-onevision": "kavaliha/llava-onevision",
        "qwen3-vl": "Qwen/Qwen3-VL-Demo",
        "qwen-vl": "artificialguybr/qwen-vl",
    }
    
    def __init__(
        self,
        space_id: str = "llava-onevision",
        timeout: int = 180
    ):
        """
        Initialize Gradio Space VLM interface.
        
        Args:
            space_id: Space identifier from AVAILABLE_SPACES or full space path
            timeout: Request timeout in seconds (default: 180)
        """
        # Import here to provide better error message
        try:
            from gradio_client import Client
        except ImportError:
            raise ImportError(
                "gradio_client is required. Install it with: pip install gradio_client"
            )
        
        # Resolve space ID
        if space_id in self.AVAILABLE_SPACES:
            self.space = self.AVAILABLE_SPACES[space_id]
            self.model_name = space_id
        else:
            self.space = space_id
            self.model_name = space_id.split('/')[-1]
        
        self.timeout = timeout
        
        print(f"Connecting to HuggingFace Space: {self.space}")
        try:
            self.client = Client(self.space)
            print(f"✓ Connected to {self.model_name}")
        except Exception as e:
            print(f"✗ Failed to connect to {self.space}")
            raise ConnectionError(f"Cannot connect to Gradio Space: {e}")
    
    def _prepare_image(self, image: Union[str, Path, Image.Image]) -> str:
        """
        Prepare image for Gradio API (returns file path).
        
        Args:
            image: Image file path or PIL Image object
            
        Returns:
            Path to image file
        """
        if isinstance(image, (str, Path)):
            return str(image)
        
        elif isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            max_size = 2048
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".jpg", 
                delete=False
            )
            image.save(temp_file.name, format="JPEG", quality=95)
            return temp_file.name
        
        else:
            raise TypeError("Image must be file path or PIL Image")
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Analyze an inscription image with a prompt.
        
        Args:
            image: Image file path or PIL Image object
            prompt: Text prompt for the model
            temperature: Sampling temperature (0.0-1.0) - not used by Gradio
            max_tokens: Maximum tokens to generate - not used by Gradio
            stream: Whether to stream the response - not used by Gradio
            timeout: Override default timeout for this request
            
        Returns:
            Dictionary with response and metadata (same format as LlamaVision)
        """
        image_path = self._prepare_image(image)
        start_time = time.time()
        
        try:
            # Call Gradio Space API
            result = self.client.predict(
                message={
                    "text": prompt,
                    "files": [image_path]
                },
                api_name="/chat"
            )
            
            elapsed = time.time() - start_time
            
            # Extract response text
            if isinstance(result, dict):
                response_text = result.get('text', str(result))
            else:
                response_text = str(result)
            
            return {
                "success": True,
                "response": response_text.strip(),
                "model": self.model_name,
                "prompt": prompt,
                "done": True,
                "total_duration": elapsed,
                "eval_count": len(response_text.split()),  # Approximate
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            
            # Check for timeout
            if "timeout" in error_msg.lower() or elapsed >= (timeout or self.timeout):
                error_msg = f"Request timed out after {elapsed:.1f} seconds. The Space may be loading or busy."
            
            return {
                "success": False,
                "error": error_msg,
                "model": self.model_name,
                "prompt": prompt,
                "done": False,
            }
    
    def analyze_multiple_images(
        self,
        images: List[Union[str, Path, Image.Image]],
        prompt: str,
        temperature: float = 0.1,
        combine_method: str = "separate"
    ) -> List[Dict[str, any]]:
        """
        Analyze multiple images of the same inscription.
        
        Args:
            images: List of image file paths or PIL Image objects
            prompt: Text prompt for the model
            temperature: Sampling temperature
            combine_method: "separate" (analyze each) or "compare" (analyze together)
            
        Returns:
            List of analysis results
        """
        if combine_method == "separate":
            results = []
            for i, image in enumerate(images):
                print(f"  Analyzing image {i+1}/{len(images)}...")
                result = self.analyze_image(image, prompt, temperature)
                result["image_index"] = i
                results.append(result)
            return results
        
        elif combine_method == "compare":
            # For comparison, analyze first image with enhanced prompt
            enhanced_prompt = (
                f"{prompt}\n\n"
                f"Note: This is one of {len(images)} photographs of the same inscription. "
                f"Focus on the most clearly visible features."
            )
            result = self.analyze_image(images[0], enhanced_prompt, temperature)
            result["num_images_provided"] = len(images)
            result["combined_method"] = "compare"
            return [result]
        
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
    
    def check_availability(self) -> bool:
        """
        Check if the Gradio Space is available and responding.
        
        Returns:
            True if space is available, False otherwise
        """
        try:
            # Try to get API info
            api_info = self.client.view_api()
            print(f"✓ {self.model_name} Space is available and responding")
            return True
            
        except Exception as e:
            print(f"✗ Cannot connect to {self.model_name} Space")
            print(f"  Error: {e}")
            print(f"\nThe Space may be:")
            print(f"  1. Starting up (cold start)")
            print(f"  2. Temporarily unavailable")
            print(f"  3. Experiencing high traffic")
            print(f"\nTry:")
            print(f"  1. Wait a few moments and retry")
            print(f"  2. Visit: https://huggingface.co/spaces/{self.space}")
            print(f"  3. Try alternative space: GradioSpaceVLM('qwen3-vl')")
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """
        Get information about the Gradio Space.
        
        Returns:
            Dictionary with space information or None if unavailable
        """
        try:
            return {
                "space": self.space,
                "model_name": self.model_name,
                "type": "HuggingFace Gradio Space",
                "cost": "FREE (Zero GPU)",
                "available_spaces": list(self.AVAILABLE_SPACES.keys())
            }
        except Exception:
            return None
    
    @staticmethod
    def list_available_spaces():
        """Print list of available Gradio Spaces."""
        print("\nAvailable HuggingFace Gradio Spaces (FREE):")
        print("-" * 60)
        for key, space in GradioSpaceVLM.AVAILABLE_SPACES.items():
            print(f"  {key:20} -> {space}")
        print("\nUsage:")
        print('  vlm = GradioSpaceVLM(space_id="llava-onevision")')
        print("\nRecommended for ancient scripts:")
        print("  - llava-onevision (LLaVA OneVision, Zero GPU)")
        print("  - qwen3-vl (Official Qwen3-VL, very popular)")
        print("\nNote: First request may be slow (cold start)")
