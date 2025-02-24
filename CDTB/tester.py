from parser_utils import parse_input, DEFAULT_GRAMMAR
from generators import GuidanceGenerator

GUIDANCE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
GUIDANCE_CACHE_DIR = "/home/alexanderchen04/constrained-decoding/models"
generator = GuidanceGenerator(GUIDANCE_MODEL_NAME, GUIDANCE_CACHE_DIR)

prompt = 'generate a json for a character profile in a rpg game with 5 traits'

raw, tokens = generator.generate(prompt)

print("raw: ",raw)
print("tokens:",tokens)

# Define data classes

'''
from dataclasses import dataclass
from typing import List
import re

@dataclass
class TokenInfo:
    text: str
    prob: float
    token_id: int

@dataclass
class ParsedToken:
    main_token: TokenInfo
    top_k_tokens: List[TokenInfo]

def extract_token_info(token_str: str) -> TokenInfo:
    # Extract token_id
    token_id_match = re.search(r'token_id=(\d+)', token_str)
    token_id = int(token_id_match.group(1)) if token_id_match else -1
    
    # Extract probability
    prob_match = re.search(r'prob=([0-9.e-]+)', token_str)
    prob = float(prob_match.group(1)) if prob_match else 0.0
    
    # Extract text
    text_match = re.search(r"text='([^']*)'", token_str)
    text = text_match.group(1) if text_match else ''
    
    return TokenInfo(text=text, prob=prob, token_id=token_id)

def parse_token_string(token_str: str) -> List[ParsedToken]:
    parsed_tokens = []
    target_sequence = ["<|im_start|>", "assistant", "\n"]
    sequence_buffer = []
    output_started = False
    
    # Find all GenTokenExtra blocks
    token_extra_pattern = r'GenTokenExtra\([^)]+top_k=\[(.*?)\]\)'
    token_extras = re.finditer(token_extra_pattern, token_str, re.DOTALL)
    
    for token_extra_match in token_extras:
        full_token_extra = token_extra_match.group(0)
        top_k_section = token_extra_match.group(1)
        
        # Parse main token
        main_token = extract_token_info(full_token_extra)
        cleaned_text = main_token.text.replace('\\n', '\n')
        
        if not output_started:
            sequence_buffer.append(cleaned_text)
 
            if len(sequence_buffer) > len(target_sequence):
                sequence_buffer.pop(0)
            if sequence_buffer == target_sequence:
                output_started = True
                sequence_buffer = []
            continue

        # Parse top_k tokens for actual output
        top_k_tokens = []
        top_k_pattern = r'GenToken\(([^)]+)\)'
        top_k_matches = re.finditer(top_k_pattern, top_k_section)
        
        for top_k_match in top_k_matches:
            top_k_token = extract_token_info(top_k_match.group(0))
            top_k_tokens.append(top_k_token)
            
        parsed_tokens.append(ParsedToken(main_token=main_token, top_k_tokens=top_k_tokens))
    
    return parsed_tokens

def format_token_info(token_info: TokenInfo, indent: int = 0) -> str:
    indent_str = " " * indent
    text_repr = repr(token_info.text.replace('\n', '\\n'))
    return f"{indent_str}Text: {text_repr}, Probability: {token_info.prob:.4f}, Token ID: {token_info.token_id}"

def print_parsed_tokens(parsed_tokens: List[ParsedToken]):
    for i, token in enumerate(parsed_tokens, 1):
        print(f"\nToken {i}:")
        print(format_token_info(token.main_token, indent=2))
        
        print("  Top-k alternatives:")
        for j, top_k in enumerate(token.top_k_tokens, 1):
            print(format_token_info(top_k, indent=4))

def read_token_string_from_file(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read().strip()

# Example usage:
if __name__ == "__main__":
    token_str = read_token_string_from_file('text.txt')
    parsed_tokens = parse_token_string(token_str)
    print_parsed_tokens(parsed_tokens)
    '''