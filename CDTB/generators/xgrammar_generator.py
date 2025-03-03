from .base_generator import BaseGenerator
import xgrammar as xgr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class  XGrammarGenerator(BaseGenerator):
    """
    A class to encapsulate grammar-based text generation using a Hugging Face model and tokenizer.
    """
    def __init__(self, model_id: str, cache_dir: str, ebnf_path=None, ebnf_in=None):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.ebnf_path = ebnf_path
        self.ebnf_in = ebnf_in
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Compile grammar
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.config = AutoConfig.from_pretrained(model_id)
        self.grammar_compiler = self._initialize_grammar_compiler()
        self.compiled_grammar = self._compile_grammar()
        self.logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)

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
        if self.ebnf_path is not None:
            ebnf_string = load_ebnf(self.ebnf_path)
        else:
            ebnf_string = self.ebnf_in
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
        model_inputs = self.prepare_model_inputs(system_prompt, user_prompt)
        logits_processor = xgr.contrib.hf.LogitsProcessor(self.compiled_grammar)

        generation_output = self.model.generate(
            **model_inputs, 
            max_new_tokens=512, 
            logits_processor=[logits_processor], 
            return_dict_in_generate=True,  # Required to get scores
            output_scores=True
        )

        generated_sequences = generation_output.sequences
        scores = generation_output.scores
        
        # Get just the generated part (excluding input)
        generated_tokens = [
            seq[len(input_ids):] for seq, input_ids in zip(generated_sequences, model_inputs.input_ids)
        ]
        
        # Decode the generated text
        responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Return both the text response and the parsed token analysis
        return (responses[0], self.process_token(generated_tokens, scores))
        """ generated_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)"""


    def process_token(self, generated_tokens, scores):
        # We'll work with just the first batch
        tokens = generated_tokens[0]
        parsed_tokens = []
        
        for step_idx, (token_id, step_scores) in enumerate(zip(tokens, scores)):
            token_text = self.tokenizer.decode([token_id])
            
            # Get scores for the first batch
            batch_scores = step_scores[0]
            
            # Get the probability of the selected token
            log_softmax = torch.nn.functional.log_softmax(batch_scores, dim=0)
            selected_logprob = log_softmax[token_id].item()
            selected_prob = torch.exp(torch.tensor(selected_logprob)).item()  # Convert logprob back to probability
    

            # Filter out -inf scores and get valid tokens
            valid_mask = batch_scores > float('-inf')
            valid_scores = batch_scores[valid_mask]
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)

            valid_log_probs = log_softmax[valid_mask]
            valid_original_probs = torch.exp(valid_log_probs)

            # Prepare the top_k_tokens list
            top_k_tokens_info = []
            
            if len(valid_indices) > 0:
                # Apply softmax to convert logits to probabilities
                #valid_probs = torch.softmax(valid_scores, dim=0)
                
                # Get as many valid tokens as available (up to a reasonable limit)
                k = min(5, len(valid_indices))
                top_k_values, top_k_idx = torch.topk(valid_original_probs, k=k)
                top_k_indices = valid_indices[top_k_idx]
                top_k_tokens = self.tokenizer.batch_decode(top_k_indices.unsqueeze(-1))
                
                # Create dictionary objects for top_k tokens
                for tok, logprob, idx in zip(top_k_tokens, top_k_values.tolist(), top_k_indices.tolist()):
                    token_info = {
                        "text": tok,
                        "prob": logprob,
                        "token_id": idx
                    }
                    top_k_tokens_info.append(token_info)
            
            # Create the parsed token dictionary
            parsed_token = {
                "main_token": {
                    "text": token_text,
                    "prob": selected_prob,
                    "token_id": token_id.item() if hasattr(token_id, 'item') else token_id
                },
                "top_k_tokens": top_k_tokens_info
            }
            
            parsed_tokens.append(parsed_token)
        
        return parsed_tokens[:-1]


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