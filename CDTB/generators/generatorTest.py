from xgrammar_generator import XGrammarGenerator

if __name__ == "__main__":
    # Configuration
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" # "meta-llama/Llama-3.2-3B-Instruct"
    CACHE_DIR = "/home/alexanderchen04/constrained-decoding/models"
    EBNF_PATH = "/home/alexanderchen04/constrained-decoding/ConstrainedCodeGen/grammars/json.gbnf"
    NUM_TESTS = 1

    # Initialize the generator
    generator = XGrammarGenerator(MODEL_ID, CACHE_DIR, EBNF_PATH)

    # Prompts for generation
    SYSTEM_PROMPT = "You are a helpful coding assistant."
    USER_PROMPT = "Generate a JSON of a person with name, gender and age"

    # Generate text
    for test in range(NUM_TESTS):
        print(f"Test #{test+1}:")
        generated_code = generator.generate(SYSTEM_PROMPT, USER_PROMPT)
        print(generated_code)
