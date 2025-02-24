from .generators.guidance_generator import GuidanceGenerator
from .generators.xgrammar_generator import XGrammarGenerator
from .parser_utils import parse_input

# Instantiate generators
generators = {
    "guidance": GuidanceGenerator(),
    "xgrammar": XGrammarGenerator(),
}

def generate_response(data):
    """Handles the logic of selecting the right generator and parsing output."""
    library = data.get("library", "guidance")  # Default to guidance
    grammar = data.get("grammar", "DEFAULT_GRAMMAR")
    text_input = data.get("input", "1")
    prompt = data.get("prompt", "Generate a JSON for an RPG character.")

    # Validate the requested library
    if library not in generators:
        return {"status": "error", "message": f"Unsupported library: {library}"}, 400

    # Generate output
    raw_output, token_output = generators[library].generate(prompt)

    # Parse output
    result = parse_input(raw_output, grammar)

    return {**result, "tokenOutput": token_output}, (200 if result['status'] == 'success' else 400)
