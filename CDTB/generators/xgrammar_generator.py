from .base_generator import BaseGenerator
import xgrammar as xgr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import time

class  XGrammarGenerator(BaseGenerator):
    """
    A class to encapsulate grammar-based text generation using a Hugging Face model and tokenizer.
    """
    def __init__(self, model_id: str, cache_dir: str, ebnf_path: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.ebnf_path = ebnf_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("grammar")
        # Compile grammar
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        print("grammar compile")
        self.grammar_compiler = self._initialize_grammar_compiler()
        print("compiled grammar")
        self.compiled_grammar = self._compile_grammar()
        self.logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)


        print("intialize model")
        # Initialize model, tokenizer, and configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, device_map="auto", torch_dtype=torch.float16, #device_map={"": 0}
        )


    def _initialize_grammar_compiler(self) -> xgr.GrammarCompiler:
        """
        Initialize the grammar compiler with tokenizer info.
        """
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.tokenizer, vocab_size=self.config.vocab_size
        )
        return xgr.GrammarCompiler(tokenizer_info)

    def _compile_grammar(self) -> xgr.Grammar:
        """
        Compile grammar using the provided EBNF file.
        """
        ebnf_string = load_ebnf(self.ebnf_path)
        print("finish load")
        return self.grammar_compiler.compile_grammar(ebnf_string)

    def prepare_model_inputs(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Prepare model inputs for generation based on system and user prompts.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        texts = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer(texts, return_tensors="pt").to(self.model.device)
        return model_inputs

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """
        Generate text based on prompts and compiled grammar.
        """
        print("model outputs")
        model_inputs = self.prepare_model_inputs(system_prompt, user_prompt)
        print("compiling grammar")
        logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)
        gen_start = time.time()
        
        print("generating")
        """with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                logits_processor=[logits_processor]
            )"""
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[logits_processor]
        )
        print(f"Generation time: {time.time() - gen_start:.2f}s")
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

def load_ebnf(file_path):
    """
    Loads the content of an ebnf file into a string with proper formatting.
    
    Args:
        file_path (str): The path to the text file.

    Returns:
        str: The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content.strip()  # Removes any leading or trailing whitespace
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""