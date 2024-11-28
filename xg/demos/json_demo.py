# CUDA_VISIBLE_DEVICES=1 python -m xg.demos.json_demo

from xg.generator import GrammarBasedGenerator

if __name__ == "__main__":
    # Configuration
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    CACHE_DIR = "/data2/.shared_models"
    EBNF_PATH = "./grammars/json.gbnf"
    NUM_TESTS = 3

    # Initialize the generator
    generator = GrammarBasedGenerator(MODEL_ID, CACHE_DIR, EBNF_PATH)

    # Prompts for generation
    SYSTEM_PROMPT = "You are a helpful coding assistant."
    USER_PROMPT = "Introduce yourself in JSON."

    # Generate text
    for test in range(NUM_TESTS):
        print(f"Test #{test+1}:")
        generated_code = generator.generate(SYSTEM_PROMPT, USER_PROMPT)
        print(generated_code)
