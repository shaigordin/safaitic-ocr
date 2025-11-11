"""
Prompt templates for analyzing Safaitic inscriptions with VLMs.

This module contains various prompt strategies optimized for different
aspects of Safaitic inscription analysis.
"""

from typing import Dict, Optional


class SafaiticPrompts:
    """
    Collection of prompt templates for Safaitic inscription analysis.
    """
    
    @staticmethod
    def basic_description() -> str:
        """Basic image description prompt."""
        return """Describe what you see in this image in detail. 
Focus on any text, inscriptions, or writing visible on the stone surface."""
    
    @staticmethod
    def script_identification() -> str:
        """Prompt for identifying the script type."""
        return """Analyze this image carefully. 

1. Is there any text or inscription visible?
2. If yes, what writing system or script does it appear to be?
3. Describe the characteristics of the characters you see.
4. Is the text carved, painted, or written in some other way?

Be specific about what you observe."""
    
    @staticmethod
    def character_recognition(provide_context: bool = True) -> str:
        """Prompt for recognizing individual characters."""
        context = ""
        if provide_context:
            context = "This is a Safaitic inscription (ancient North Arabian script, right-to-left, 28 consonantal letters).\n\n"
        
        return f"""{context}List the characters you can identify in this inscription:
1. Individual letters or characters visible
2. Writing direction
3. Any unclear or damaged portions"""
    
    @staticmethod
    def transliteration_attempt(include_primer: bool = True) -> str:
        """Prompt for attempting transliteration."""
        primer = ""
        if include_primer:
            primer = "Safaitic letters: ʾ b g d h w z ḥ ṭ y k l m n s ʿ f ṣ q r t ṯ ḫ ḏ ḍ ġ. Often starts with 'l' (by/of) + name.\n\n"
        
        return f"""{primer}Transliterate this Safaitic inscription (right-to-left):
1. Identify visible characters
2. Provide Latin transliteration
3. Mark uncertain readings with [?]
4. Note damaged portions"""
    
    @staticmethod
    def translation_attempt() -> str:
        """Prompt for attempting translation."""
        return """This is a Safaitic inscription by ancient nomads.

Common patterns: "l + name" (by [name]), "bn" (son of), actions (camping, herding, raiding, grieving).

Provide:
1. General structure
2. Personal names (if visible)
3. Formulaic phrases
4. Possible translation (mark uncertainties)"""
    
    @staticmethod
    def comparative_analysis() -> str:
        """Prompt for comparing multiple images of same inscription."""
        return """You are viewing multiple photographs of the same ancient Safaitic inscription taken from different angles or lighting conditions.

Analyze all images and:
1. Identify characters that are clearer in some images than others
2. Note consistent features visible across all images
3. Highlight any characters that appear different due to perspective/lighting
4. Provide your best reading of the inscription based on all images combined
5. Indicate which image(s) were most helpful for reading specific portions

Synthesize information from all images for the most accurate reading."""
    
    @staticmethod
    def condition_assessment() -> str:
        """Prompt for assessing inscription condition."""
        return """Assess the physical condition of this Safaitic inscription:

1. Surface condition: weathering, erosion, damage
2. Clarity: how clear are the characters?
3. Completeness: is the inscription complete or broken?
4. Readability factors: lighting, angle, surface texture
5. Any modern interference: graffiti, wear from touching, etc.

Rate the overall legibility on a scale of 1-5:
1 = Nearly illegible
2 = Poorly preserved, few characters clear
3 = Moderately preserved, some characters readable
4 = Well preserved, most characters clear
5 = Excellent condition, all characters clear

Provide specific observations supporting your assessment."""
    
    @staticmethod
    def guided_analysis(
        transliteration: Optional[str] = None,
        translation: Optional[str] = None
    ) -> str:
        """Prompt with ground truth for evaluation."""
        parts = ["Analyze this Safaitic inscription carefully."]
        
        if transliteration:
            parts.append(f"\nExpected transliteration: {transliteration}")
            parts.append("\nCan you identify these characters in the image?")
            parts.append("Mark which characters you can clearly see vs. which are unclear.")
        
        if translation:
            parts.append(f"\nExpected meaning: {translation}")
            parts.append("\nBased on this translation, can you identify:")
            parts.append("- The personal name(s)")
            parts.append("- Genealogical markers (son of, etc.)")
            parts.append("- Action verbs or content words")
        
        return "\n".join(parts)
    
    @staticmethod
    def structured_json_output() -> str:
        """Prompt requesting structured JSON output."""
        return """Analyze this Safaitic inscription and provide your analysis in the following JSON format:

{
  "script_identified": true/false,
  "script_type": "Safaitic" or other,
  "direction": "right-to-left" or other,
  "characters_visible": ["list", "of", "characters"],
  "transliteration": "your transliteration or null",
  "translation": "your translation or null",
  "personal_names": ["identified", "names"],
  "confidence": {
    "script_identification": 0-100,
    "transliteration": 0-100,
    "translation": 0-100
  },
  "condition": {
    "legibility_score": 1-5,
    "preservation": "excellent/good/fair/poor",
    "issues": ["weathering", "damage", "etc"]
  },
  "notes": "any additional observations"
}

Provide only valid JSON, no other text."""
    
    @staticmethod
    def custom_prompt(
        task: str,
        context: str = "",
        constraints: str = "",
        output_format: str = ""
    ) -> str:
        """Create a custom prompt from components."""
        parts = []
        
        if context:
            parts.append(f"Context: {context}\n")
        
        parts.append(f"Task: {task}")
        
        if constraints:
            parts.append(f"\nConstraints: {constraints}")
        
        if output_format:
            parts.append(f"\nOutput format: {output_format}")
        
        return "\n".join(parts)
    
    @classmethod
    def get_all_prompts(cls) -> Dict[str, str]:
        """Get dictionary of all available standard prompts."""
        return {
            "basic_description": cls.basic_description(),
            "script_identification": cls.script_identification(),
            "character_recognition": cls.character_recognition(),
            "transliteration_attempt": cls.transliteration_attempt(),
            "translation_attempt": cls.translation_attempt(),
            "comparative_analysis": cls.comparative_analysis(),
            "condition_assessment": cls.condition_assessment(),
            "structured_json": cls.structured_json_output(),
        }
