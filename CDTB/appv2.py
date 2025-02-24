from flask import Flask, request, jsonify
from flask_cors import CORS
#from parsers.parser_utils import parse_input, DEFAULT_GRAMMAR
#from generators.guidance_generator import GuidanceGenerator

from parser_utils import parse_input, DEFAULT_GRAMMAR
from generators import GuidanceGenerator

app = Flask(__name__)
CORS(app)

# Initialize the generator
GUIDANCE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
GUIDANCE_CACHE_DIR = "/home/alexanderchen04/constrained-decoding/models"
generator = GuidanceGenerator(GUIDANCE_MODEL_NAME, GUIDANCE_CACHE_DIR)

@app.route('/parse', methods=['POST'])
def parse():
    """Handles the parsing request."""
    data = request.json
    grammar = data.get('grammar', DEFAULT_GRAMMAR)
    text_input = data.get('input', "1")
    prompt = data.get('prompt', 'generate a json for a character profile in a rpg game with 5 traits')

    # Generate output using the generator
    raw_output, token_output = generator.generate(prompt)

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