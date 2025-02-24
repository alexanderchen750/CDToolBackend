from flask import Flask, request, jsonify
from flask_cors import CORS
#from parsers.parser_utils import parse_input, DEFAULT_GRAMMAR
#from generators.guidance_generator import GuidanceGenerator

from parser_utils import parse_input, DEFAULT_GRAMMAR
from generators import GuidanceGenerator, XGrammarGenerator

app = Flask(__name__)
CORS(app)

# Initialize the generator
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_CACHE_DIR = "/home/alexanderchen04/constrained-decoding/models"
#generator = GuidanceGenerator(MODEL_NAME, MODEL_CACHE_DIR)
EBNF_PATH = "/home/alexanderchen04/constrained-decoding/ConstrainedCodeGen/grammars/json.gbnf"
SYSTEM_PROMPT = "You are a system that generates what the user asks for in the correct format they ask for"

@app.route('/parse', methods=['POST'])
def parse():
    """Handles the parsing request."""
    data = request.json
    library = data.get('library', "xgrammar")
    grammar = data.get('grammar', DEFAULT_GRAMMAR)
    text_input = data.get('input', "1")
    prompt = data.get('prompt', 'generate a json for a character profile in a rpg game with 5 traits')

    # Generate output using the generator
    if library == "guidance":
        generator = GuidanceGenerator(MODEL_NAME, MODEL_CACHE_DIR)
        raw_output, token_output = generator.generate(prompt)

    if library == "xgrammar":
        xGrammarGenerator = XGrammarGenerator(MODEL_NAME,MODEL_CACHE_DIR,EBNF_PATH)
        raw_output = xGrammarGenerator.generate(SYSTEM_PROMPT,prompt)
        token_output=""
        print("XGRAMMAR")



    # Parse the model output
    result = parse_input(raw_output, grammar)

    # Return the result as JSON
    return jsonify({**result, "tokenOutput": token_output}), (200 if result['status'] == 'success' else 400)

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to check if the server is running."""
    return jsonify({"message": "test"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)