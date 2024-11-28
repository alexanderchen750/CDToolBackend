import xgrammar as xgr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from xg.utils import load_ebnf


class GrammarBasedGenerator:
    """
    A class to encapsulate grammar-based text generation using a Hugging Face model and tokenizer.
    """

    def __init__(self, model_id: str, cache_dir: str, ebnf_path: str):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.ebnf_path = ebnf_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Compile grammar
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        self.grammar_compiler = self._initialize_grammar_compiler()
        self.compiled_grammar = self._compile_grammar()

        # Initialize model, tokenizer, and configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, device_map="auto"
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

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generate text based on prompts and compiled grammar.
        """
        model_inputs = self.prepare_model_inputs(system_prompt, user_prompt)
        logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            logits_processor=[logits_processor]
        )
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
