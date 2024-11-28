# CUDA_VISIBLE_DEVICES=1 python outlines/example.py

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize Model + Tokenizer
model_id  = "meta-llama/Llama-3.2-3B-Instruct"
cache_dir = "/data2/.shared_models"
llm       = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = outlines.models.Transformers(llm, tokenizer)

# Initialize Grammar
arithmetic_grammar = """
    ?start: expression

    ?expression: term (("+" | "-") term)*

    ?term: factor (("*" | "/") factor)*

    ?factor: NUMBER
           | "-" factor
           | "(" expression ")"

    %import common.NUMBER
"""

# Generate with Grammar
generator = outlines.generate.cfg(model, arithmetic_grammar)
sequence = generator("Alice had 4 apples and Bob ate 2. Write an expression for Alice's apples:")

print(sequence)