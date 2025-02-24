from flask import Flask, request, jsonify
from flask_cors import CORS
from parser_utils import parse_input, DEFAULT_GRAMMAR
from guidance_module.guidance_functions import gen

app = Flask(__name__)
CORS(app)

@app.route('/parse', methods=['POST'])
def parse():
    """Handles the parsing request."""
    data = request.json
    grammar = data.get('grammar', DEFAULT_GRAMMAR)  # Use provided grammar or default
    text_input = data.get('input', "1")
    prompt = data.get('prompt', 'generate a json for a character profile in a rpg game with 5 traits')

    # Run the model to generate an output (for now using text_input directly)
    #model_output = text_input  # Replace with: 
    model_output = gen(prompt)

    # Parse the model output
    result = parse_input(model_output, grammar)

    # Return the result as JSON
    return jsonify(result), (200 if result['status'] == 'success' else 400)

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to check if the server is running."""
    return jsonify({"message": "test"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
