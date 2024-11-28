# CUDA_VISIBLE_DEVICES=1 python -m xg.demos.c_demo

from xg.generator import GrammarBasedGenerator

if __name__ == "__main__":
    # Configuration
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" # "meta-llama/Llama-3.2-3B-Instruct"
    CACHE_DIR = "/data2/.shared_models"
    EBNF_PATH = "./grammars/c.gbnf"
    NUM_TESTS = 3

    # Initialize the generator
    generator = GrammarBasedGenerator(MODEL_ID, CACHE_DIR, EBNF_PATH)

    # Prompts for generation
    SYSTEM_PROMPT = "You are a helpful coding assistant."
    USER_PROMPT = "Write a unintuitive C implementation of the Fibonacci function."

    # Generate text
    for test in range(NUM_TESTS):
        print(f"Test #{test+1}:")
        generated_code = generator.generate(SYSTEM_PROMPT, USER_PROMPT)
        print(generated_code)
