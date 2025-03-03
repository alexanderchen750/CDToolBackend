import outlines
from .base_generator import BaseGenerator
import torch

class  OutlinesGenerator(BaseGenerator):
    """
    A class to encapsulate grammar-based text generation using a Hugging Face model and tokenizer.
    """
    def __init__(self, model_id: str, grammar: str):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = outlines.models.transformers(self.model_id)
        self.grammar = grammar

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate text based on prompts and compiled grammar.
        """
        temp_prompt = "your are a helpful assistant."

        prompt = f"""
        <|im_start|>system
        {temp_prompt}
        <|im_end|>

        <|im_start|>user
        {user_prompt}
        <|im_end|>
        <|im_start|>assistant
        """
        print(prompt)
        generator = outlines.generate.cfg(self.model, self.grammar)
        return generator(prompt) 
        