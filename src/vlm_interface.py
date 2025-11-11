"""
Interface for Llama 3.2 Vision model via Ollama.

This module provides a wrapper for interacting with Llama 3.2 Vision model
for analyzing Safaitic inscriptions.
"""

import ollama
from pathlib import Path
from typing import Union, List, Dict, Optional
from PIL import Image
import io
import time


class LlamaVision:
    """
    Interface for LlamaVision model running on local Ollama server.
    This is a FREE alternative that runs entirely on your local machine.
    
    Example:
        >>> vlm = LlamaVision()
        >>> if vlm.check_availability():
        ...     result = vlm.analyze_image("path/to/image.jpg", "What is in this image?")
        ...     print(result["response"])
    """
    
    def __init__(
        self,
        model_name: str = "llama3.2-vision",
        timeout: int = 300
    ):
        """
        Initialize Llama Vision interface using official Ollama library.
        
        Args:
            model_name: Name of the Ollama model (e.g., "llama3.2-vision", "llava", "qwen3-vl")
            timeout: Request timeout in seconds (default: 300 for vision models)
        """
        self.model_name = model_name
        self.timeout = timeout
        self.client = ollama.Client(timeout=timeout)
    
    def _prepare_image(self, image: Union[str, Path, Image.Image]) -> Union[str, Path]:
        """
        Prepare image for Ollama API.
        
        Args:
            image: Image file path or PIL Image object
            
        Returns:
            Path to image file (Ollama handles file reading internally)
        """
        if isinstance(image, (str, Path)):
            return str(image)
        elif isinstance(image, Image.Image):
            # Save PIL Image to temp file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(temp_file.name, format='JPEG', quality=95)
            return temp_file.name
        else:
            raise TypeError("Image must be file path or PIL Image")
    
    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, any]:
        """
        Analyze an inscription image with a prompt using Ollama's official library.
        
        Args:
            image: Image file path or PIL Image object
            prompt: Text prompt for the model
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response (not used, kept for API compatibility)
            
        Returns:
            Dictionary with response and metadata
        """
        image_path = self._prepare_image(image)
        
        options = {
            "temperature": temperature,
        }
        
        if max_tokens:
            options["num_predict"] = max_tokens
        
        start_time = time.time()
        
        try:
            # Use official Ollama client with vision model
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_path],
                options=options,
                stream=False
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract response text - response is a Pydantic GenerateResponse object
            # Access attributes directly, not as dict
            response_text = response.response if hasattr(response, 'response') else str(response)
            total_duration_ns = getattr(response, 'total_duration', None)
            
            # Convert nanoseconds to seconds, handling None
            if total_duration_ns is not None and total_duration_ns > 0:
                total_duration = total_duration_ns / 1_000_000_000
            else:
                total_duration = elapsed_time
            
            return {
                "success": True,
                "response": response_text,
                "model": self.model_name,
                "prompt": prompt,
                "elapsed_time": elapsed_time,
                "total_duration": total_duration,
            }
            
        except Exception as e:
            error_msg = str(e)
            # Check for timeout
            if "timeout" in error_msg.lower():
                error_msg = f"Request timed out after {self.timeout} seconds. Try increasing timeout or using a simpler prompt."
            
            return {
                "success": False,
                "error": error_msg,
                "model": self.model_name,
                "prompt": prompt,
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
                print(f"Analyzing image {i+1}/{len(images)}...")
                result = self.analyze_image(image, prompt, temperature)
                result["image_index"] = i
                results.append(result)
            return results
        
        elif combine_method == "compare":
            # For multi-image comparison, prepare all images
            image_paths = [self._prepare_image(img) for img in images]
            
            enhanced_prompt = f"{prompt}\n\nYou are analyzing {len(images)} different photographs of the same inscription."
            
            options = {"temperature": temperature}
            
            try:
                start_time = time.time()
                response = self.client.generate(
                    model=self.model_name,
                    prompt=enhanced_prompt,
                    images=image_paths,
                    options=options,
                    stream=False
                )
                
                elapsed_time = time.time() - start_time
                
                if isinstance(response, dict):
                    response_text = response.get("response", "")
                else:
                    response_text = str(response)
                
                return [{
                    "success": True,
                    "response": response_text,
                    "model": self.model_name,
                    "prompt": enhanced_prompt,
                    "num_images": len(images),
                    "elapsed_time": elapsed_time,
                }]
                
            except Exception as e:
                return [{
                    "success": False,
                    "error": str(e),
                    "model": self.model_name,
                    "prompt": enhanced_prompt,
                }]
        
        else:
            raise ValueError(f"Unknown combine_method: {combine_method}")
    
    def check_availability(self) -> bool:
        """
        Check if the model is available and responding using Ollama client.
        
        Returns:
            True if model is available, False otherwise
        """
        try:
            response = self.client.list()
            
            # The response has .models attribute with Model objects
            if hasattr(response, 'models'):
                models_list = response.models
            else:
                models_list = []
            
            # Extract model names (could be "llama3.2-vision:latest" or just "llama3.2-vision")
            available_models = []
            for m in models_list:
                if hasattr(m, 'model'):
                    available_models.append(m.model)
                elif isinstance(m, dict):
                    available_models.append(m.get('name', ''))
            
            # Check if our model is in the list (with or without :latest suffix)
            available = any(
                self.model_name in model or model.startswith(self.model_name + ':')
                for model in available_models
            )
            
            if available:
                print(f"✓ {self.model_name} is available")
            else:
                print(f"✗ {self.model_name} not found. Available models:")
                for model in available_models:
                    print(f"  - {model}")
                print(f"\nTo install: ollama pull {self.model_name}")
            
            return available
            
        except Exception as e:
            print(f"✗ Cannot connect to Ollama")
            print(f"  Error: {e}")
            print(f"\nMake sure Ollama is running:")
            print(f"  1. Install: https://ollama.ai")
            print(f"  2. Run: ollama serve")
            print(f"  3. Pull model: ollama pull {self.model_name}")
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """
        Get information about the loaded model using Ollama client.
        
        Returns:
            Dictionary with model information or None if unavailable
        """
        try:
            info = self.client.show(self.model_name)
            return info
        except Exception:
            return None
