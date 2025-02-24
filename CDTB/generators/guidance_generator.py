import guidance
from guidance import models, json, user, assistant, capture
from .base_generator import BaseGenerator
import re
import ast
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json as json_lib  # renamed to avoid conflict with guidance.json


@dataclass
class TokenInfo:
    text: str
    prob: float
    token_id: int

@dataclass
class ParsedToken:
    main_token: TokenInfo
    top_k_tokens: List[TokenInfo]

class GuidanceGenerator(BaseGenerator):
    def __init__(self, model_name: str, cache_dir: str):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = models.Transformers(model_name, cache_dir=cache_dir)

    @guidance
    def _gen_json_internal(self, lm, prompt: str):
        """Internal guidance function to generate JSON output."""
        with user():
            lm += prompt
        with assistant():
            lm += json()
        return lm

    def _trim_output(self, output: str):
        """Clean up the output from the model."""
        match = re.search(r"<\|im_start\|>assistant\s(.*)", output, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError("Failed to clean output: Invalid format")

    def generate(self, prompt: str, **kwargs):
        """Generate JSON output using the guidance library."""
        if not prompt:
            prompt = "Generate 10 token response of anything you want"

        # Generate output using the guidance model
        lm = self.model + capture(self._gen_json_internal(prompt), "response")
        token_stats = lm.get_per_token_stats()
        raw_output = lm["response"]

        if hasattr(token_stats, '__str__'):
            token_str = str(token_stats)
        else:
            # If the object doesn't have a string representation, we'll need to process it differently
            token_str = self._process_token_stats(token_stats)

        # Clean and return the output
        return (self._trim_output(raw_output),self._parse_tokens(token_str))
        #return self._trim_output(raw_output)

    def _extract_token_info(self, token_str: str) -> TokenInfo:
        """Extract token information from a token string."""
        token_id_match = re.search(r'token_id=(\d+)', token_str)
        token_id = int(token_id_match.group(1)) if token_id_match else -1
        
        prob_match = re.search(r'prob=([0-9.e-]+)', token_str)
        prob = float(prob_match.group(1)) if prob_match else 0.0
        
        text_match = re.search(r"text='([^']*)'", token_str)
        text = text_match.group(1) if text_match else ''
        
        return TokenInfo(text=text, prob=prob, token_id=token_id)

    def _process_token_stats(self, token_stats) -> str:
        """Process token statistics from the Cache object into a string format."""
        try:
            # Try to get the token information directly
            if hasattr(token_stats, 'tokens'):
                return str(token_stats.tokens)
            elif hasattr(token_stats, 'values'):
                return str(token_stats.values)
            else:
                # If we can't get direct access, convert the whole object to string
                return str(token_stats)
        except Exception as e:
            print(f"Error processing token stats: {e}")
            print(f"Token stats type: {type(token_stats)}")
            print(f"Token stats dir: {dir(token_stats)}")
            raise ValueError(f"Unable to process token stats: {e}")

    def _parse_tokens(self, token_str: str) -> List[ParsedToken]:
        """Parse token string into a list of ParsedToken objects."""
        parsed_tokens = []
        target_sequence = ["<|im_start|>", "assistant", "\n"]
        sequence_buffer = []
        output_started = False
        
        token_extra_pattern = r'GenTokenExtra\([^)]+top_k=\[(.*?)\]\)'
        token_extras = re.finditer(token_extra_pattern, token_str, re.DOTALL)
        
        for token_extra_match in token_extras:
            full_token_extra = token_extra_match.group(0)
            top_k_section = token_extra_match.group(1)
            
            main_token = self._extract_token_info(full_token_extra)
            cleaned_text = main_token.text.replace('\\n', '\n')
            
            if not output_started:
                sequence_buffer.append(cleaned_text)
                if len(sequence_buffer) > len(target_sequence):
                    sequence_buffer.pop(0)
                if sequence_buffer == target_sequence:
                    output_started = True
                    sequence_buffer = []
                continue

            top_k_tokens = []
            top_k_pattern = r'GenToken\(([^)]+)\)'
            top_k_matches = re.finditer(top_k_pattern, top_k_section)
            
            for top_k_match in top_k_matches:
                top_k_token = self._extract_token_info(top_k_match.group(0))
                top_k_tokens.append(top_k_token)
                
            parsed_tokens.append(ParsedToken(main_token=main_token, top_k_tokens=top_k_tokens))
        
        return parsed_tokens

    def _token_to_dict(self, token: TokenInfo) -> Dict[str, Any]:
        """Convert TokenInfo to a dictionary."""
        return {
            "text": token.text,
            "probability": token.prob,
            "token_id": token.token_id
        }

    def clean_tokens(self, token_str: str) -> str:
        """
        Clean and parse token string, returning a JSON-formatted string of token information.
        
        Args:
            token_str: Raw token string from the model
            
        Returns:
            JSON string containing parsed token information with the following structure:
            {
                "tokens": [
                    {
                        "main_token": {
                            "text": str,
                            "probability": float,
                            "token_id": int
                        },
                        "alternatives": [
                            {
                                "text": str,
                                "probability": float,
                                "token_id": int
                            },
                            ...
                        ]
                    },
                    ...
                ]
            }
        """
        parsed_tokens = self._parse_tokens(token_str)
        
        output_data = {
            "tokens": [
                {
                    "main_token": self._token_to_dict(token.main_token),
                    "alternatives": [self._token_to_dict(alt) for alt in token.top_k_tokens]
                }
                for token in parsed_tokens
            ]
        }
        
        return json_lib.dumps(output_data, ensure_ascii=False, indent=2)