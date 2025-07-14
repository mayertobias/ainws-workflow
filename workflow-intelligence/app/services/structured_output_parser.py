"""
Structured Output Parser for AI Intelligence Generation

This module implements a bulletproof JSON output parser using Pydantic models
to ensure LLM responses are strictly structured and validated.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, Type, TypeVar, get_type_hints
from pydantic import BaseModel, ValidationError

from ..models.intelligence import (
    MusicalMeaningInsight, HitComparisonInsight, NoveltyAssessmentInsight,
    ProductionFeedback, StrategicInsights, AnalysisType
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class StructuredOutputParser:
    """
    Lightweight JSON output parser using only Pydantic models.
    
    This parser:
    1. Generates format instructions from Pydantic model schemas
    2. Validates all outputs against Pydantic models
    3. Provides robust error handling and fallback mechanisms
    4. Does exactly what we need, nothing more
    """
    
    def __init__(self):
        """Initialize the structured output parser."""
        self.model_mapping: Dict[AnalysisType, Type[BaseModel]] = {
            AnalysisType.MUSICAL_MEANING: MusicalMeaningInsight,
            AnalysisType.HIT_COMPARISON: HitComparisonInsight,
            AnalysisType.NOVELTY_ASSESSMENT: NoveltyAssessmentInsight,
            AnalysisType.PRODUCTION_FEEDBACK: ProductionFeedback,
            AnalysisType.STRATEGIC_INSIGHTS: StrategicInsights,
        }
        
        logger.info("Lightweight structured output parser initialized")
    
    def get_format_instructions(self, analysis_type: AnalysisType) -> str:
        """
        Generate format instructions from Pydantic model schema.
        
        Args:
            analysis_type: The type of analysis
            
        Returns:
            Format instructions string for the LLM prompt
        """
        model_class = self.model_mapping.get(analysis_type)
        if not model_class:
            raise ValueError(f"No model available for analysis type: {analysis_type}")
        
        # Get JSON schema from Pydantic model
        schema = model_class.model_json_schema()
        
        # Create format instructions
        instructions = f"""
**RESPONSE FORMAT REQUIREMENTS:**

You must return a valid JSON object that matches this exact schema:

```json
{json.dumps(schema, indent=2)}
```

**CRITICAL COMPLIANCE RULES:**
- Response MUST be ONLY a valid JSON object
- No markdown formatting, code blocks, or explanations
- All required fields must be present
- String values must be properly escaped
- Arrays must contain the specified types
- Numbers must be within valid ranges if specified

**EXAMPLE STRUCTURE:**
{self._generate_example_json(model_class)}

**VALIDATION:** Your response will be validated against this schema. Invalid responses will be rejected.
        """
        
        return instructions.strip()
    
    def _generate_example_json(self, model_class: Type[BaseModel]) -> str:
        """Generate example JSON from Pydantic model."""
        try:
            # Create a sample instance with example data
            example_data = {}
            schema = model_class.model_json_schema()
            
            for field_name, field_info in schema.get('properties', {}).items():
                field_type = field_info.get('type', 'string')
                
                if field_type == 'string':
                    example_data[field_name] = f"Example {field_name.replace('_', ' ')}"
                elif field_type == 'number':
                    example_data[field_name] = 0.75
                elif field_type == 'integer':
                    example_data[field_name] = 85
                elif field_type == 'boolean':
                    example_data[field_name] = True
                elif field_type == 'array':
                    item_type = field_info.get('items', {}).get('type', 'string')
                    if item_type == 'string':
                        example_data[field_name] = ["Example item 1", "Example item 2"]
                    else:
                        example_data[field_name] = [1, 2, 3]
                else:
                    example_data[field_name] = f"Example {field_name}"
            
            return json.dumps(example_data, indent=2)
        except Exception:
            return '{\n  "field_name": "example_value",\n  "array_field": ["item1", "item2"]\n}'
    
    def parse_structured_output(self, 
                               raw_output: str, 
                               analysis_type: AnalysisType) -> Optional[BaseModel]:
        """
        Parse and validate LLM output against the expected Pydantic model.
        
        Args:
            raw_output: Raw LLM response
            analysis_type: Type of analysis expected
            
        Returns:
            Validated Pydantic model instance or None if parsing fails
        """
        model_class = self.model_mapping.get(analysis_type)
        if not model_class:
            logger.error(f"No model available for analysis type: {analysis_type}")
            return None
        
        try:
            # Step 1: Clean and extract JSON
            cleaned_output = self._clean_and_extract_json(raw_output)
            
            # Step 2: Parse JSON
            data = json.loads(cleaned_output)
            
            # Step 3: Validate with Pydantic
            parsed_result = model_class.model_validate(data)
            
            # Step 4: Additional validation
            if not self._validate_parsed_result(parsed_result, analysis_type):
                logger.warning(f"Parsed result failed additional validation for {analysis_type}")
                return None
            
            logger.info(f"Successfully parsed {analysis_type} output")
            return parsed_result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {analysis_type}: {e}")
            return self._fallback_parse(raw_output, analysis_type)
        except ValidationError as e:
            logger.error(f"Pydantic validation error for {analysis_type}: {e}")
            return self._fallback_parse(raw_output, analysis_type)
        except Exception as e:
            logger.error(f"Unexpected error parsing {analysis_type}: {e}")
            return self._fallback_parse(raw_output, analysis_type)
    
    def _clean_and_extract_json(self, raw_output: str) -> str:
        """
        Clean and extract JSON from potentially messy LLM output.
        
        Args:
            raw_output: Raw LLM response
            
        Returns:
            Cleaned JSON string
        """
        # Remove code blocks
        import re
        
        # Pattern to match JSON code blocks
        code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
        code_block_match = re.search(code_block_pattern, raw_output, re.DOTALL)
        
        if code_block_match:
            json_content = code_block_match.group(1).strip()
        else:
            json_content = raw_output.strip()
        
        # Extract JSON object patterns
        json_patterns = [
            r'\{(?:[^{}]|{[^}]*})*\}',  # Nested object pattern
            r'\{.*?\}',                  # Simple object pattern
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, json_content, re.DOTALL)
            if json_match:
                potential_json = json_match.group().strip()
                # Validate it's actually JSON
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    continue
        
        # Return cleaned content as-is if no JSON found
        return json_content
    
    def _validate_parsed_result(self, result: BaseModel, analysis_type: AnalysisType) -> bool:
        """
        Additional validation for parsed results.
        
        Args:
            result: Parsed Pydantic model
            analysis_type: Type of analysis
            
        Returns:
            True if validation passes
        """
        # Check that required fields have meaningful content
        if analysis_type == AnalysisType.MUSICAL_MEANING:
            musical_result = result  # type: MusicalMeaningInsight
            if not musical_result.emotional_core or len(musical_result.emotional_core.strip()) < 10:
                return False
        
        elif analysis_type == AnalysisType.HIT_COMPARISON:
            hit_result = result  # type: HitComparisonInsight
            if not hit_result.market_positioning or len(hit_result.market_positioning.strip()) < 10:
                return False
        
        elif analysis_type == AnalysisType.NOVELTY_ASSESSMENT:
            novelty_result = result  # type: NoveltyAssessmentInsight
            if not novelty_result.unique_elements or len(novelty_result.unique_elements) == 0:
                return False
        
        return True
    
    def _fallback_parse(self, raw_output: str, analysis_type: AnalysisType) -> Optional[BaseModel]:
        """
        Fallback parsing when structured parsing fails.
        
        Args:
            raw_output: Raw LLM response
            analysis_type: Type of analysis
            
        Returns:
            Fallback Pydantic model with extracted content
        """
        logger.warning(f"Using fallback parsing for {analysis_type}")
        
        try:
            # Try to extract meaningful text content
            extracted_content = self._extract_text_content(raw_output)
            
            # Create fallback models with extracted content
            model_class = self.model_mapping[analysis_type]
            
            if analysis_type == AnalysisType.MUSICAL_MEANING:
                return MusicalMeaningInsight.model_validate({
                    'emotional_core': extracted_content.get('emotional_core', 'Unable to analyze emotional core'),
                    'musical_narrative': extracted_content.get('narrative', 'Musical narrative analysis unavailable'),
                    'cultural_context': extracted_content.get('cultural', 'Cultural context analysis unavailable'),
                    'listener_impact': extracted_content.get('impact', 'Listener impact assessment unavailable'),
                    'key_strengths': ['Analysis partially available from AI response'],
                    'improvement_areas': ['Structured analysis recommended']
                })
            
            elif analysis_type == AnalysisType.HIT_COMPARISON:
                return HitComparisonInsight.model_validate({
                    'hit_alignment_score': 0.5,
                    'market_positioning': extracted_content.get('positioning', 'Market positioning analysis unavailable'),
                    'target_audience': extracted_content.get('audience', 'Target audience analysis unavailable'),
                    'commercial_strengths': ['Partial analysis available'],
                    'commercial_weaknesses': ['Complete analysis recommended']
                })
            
            elif analysis_type == AnalysisType.NOVELTY_ASSESSMENT:
                return NoveltyAssessmentInsight.model_validate({
                    'innovation_score': 0.5,
                    'unique_elements': extracted_content.get('unique', ['Innovation analysis partially available']),
                    'trend_alignment': extracted_content.get('trend', 'Trend analysis unavailable'),
                    'risk_assessment': extracted_content.get('risk', 'Risk assessment unavailable'),
                    'market_readiness': extracted_content.get('readiness', 'Market readiness assessment unavailable')
                })
            
            elif analysis_type == AnalysisType.PRODUCTION_FEEDBACK:
                return ProductionFeedback.model_validate({
                    'overall_quality': extracted_content.get('quality', 'Production quality analysis unavailable'),
                    'technical_strengths': ['Partial analysis available'],
                    'technical_issues': ['Complete analysis recommended'],
                    'sonic_character': extracted_content.get('character', 'Sonic character analysis unavailable')
                })
            
            elif analysis_type == AnalysisType.STRATEGIC_INSIGHTS:
                return StrategicInsights.model_validate({
                    'market_opportunity': extracted_content.get('opportunity', 'Market opportunity analysis unavailable'),
                    'competitive_advantage': extracted_content.get('advantage', 'Competitive advantage analysis unavailable'),
                    'release_strategy': extracted_content.get('strategy', 'Release strategy recommendations unavailable'),
                    'promotional_angles': ['Strategic analysis partially available']
                })
            
            else:
                logger.error(f"No fallback available for {analysis_type}")
                return None
                
        except Exception as e:
            logger.error(f"Fallback parsing failed for {analysis_type}: {e}")
            return None
    
    def _extract_text_content(self, text: str) -> Dict[str, str]:
        """
        Extract meaningful content from unstructured text.
        
        Args:
            text: Raw text content
            
        Returns:
            Dictionary of extracted content
        """
        import re
        
        # Common patterns for extracting content
        patterns = {
            'emotional_core': [r'emotional?\s*core:?\s*([^\n]+)', r'emotion:?\s*([^\n]+)'],
            'narrative': [r'narrative:?\s*([^\n]+)', r'story:?\s*([^\n]+)'],
            'cultural': [r'cultural?\s*context:?\s*([^\n]+)', r'culture:?\s*([^\n]+)'],
            'impact': [r'impact:?\s*([^\n]+)', r'effect:?\s*([^\n]+)'],
            'positioning': [r'positioning:?\s*([^\n]+)', r'market:?\s*([^\n]+)'],
            'audience': [r'audience:?\s*([^\n]+)', r'target:?\s*([^\n]+)'],
            'unique': [r'unique:?\s*([^\n]+)', r'innovation:?\s*([^\n]+)'],
            'trend': [r'trend:?\s*([^\n]+)', r'current:?\s*([^\n]+)'],
            'risk': [r'risk:?\s*([^\n]+)', r'concern:?\s*([^\n]+)'],
            'readiness': [r'readiness:?\s*([^\n]+)', r'ready:?\s*([^\n]+)'],
            'quality': [r'quality:?\s*([^\n]+)', r'production:?\s*([^\n]+)'],
            'character': [r'character:?\s*([^\n]+)', r'sonic:?\s*([^\n]+)'],
            'opportunity': [r'opportunity:?\s*([^\n]+)', r'market:?\s*([^\n]+)'],
            'advantage': [r'advantage:?\s*([^\n]+)', r'competitive:?\s*([^\n]+)'],
            'strategy': [r'strategy:?\s*([^\n]+)', r'approach:?\s*([^\n]+)']
        }
        
        extracted = {}
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    extracted[key] = match.group(1).strip()
                    break
        
        return extracted
    
    def create_enhanced_prompt(self, base_prompt: str, analysis_type: AnalysisType) -> str:
        """
        Create an enhanced prompt with format instructions.
        
        Args:
            base_prompt: Base prompt text
            analysis_type: Type of analysis
            
        Returns:
            Enhanced prompt with format instructions
        """
        format_instructions = self.get_format_instructions(analysis_type)
        
        enhanced_prompt = f"""{base_prompt}

{format_instructions}

Remember: Your response must be ONLY a valid JSON object. No additional text or explanations."""
        
        return enhanced_prompt