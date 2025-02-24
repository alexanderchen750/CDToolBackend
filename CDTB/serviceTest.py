# test_imports.py
try:
    from generators.guidance_generator import GuidanceGenerator
    from generators.xgrammar_generator import XGrammarGenerator
    from parser_utils import parse_input

    print("✅ Imports successful!")

    # Instantiate generators
    guidance_gen = GuidanceGenerator()
    xgrammar_gen = XGrammarGenerator()

    # Run a simple test call
    test_data = {"library": "guidance", "grammar": "DEFAULT_GRAMMAR", "input": "1", "prompt": "Generate something."}
    raw_output, token_output = guidance_gen.generate(test_data["prompt"])

    # Parse the output
    result = parse_input(raw_output, test_data["grammar"])

    print(f"✅ Generator output: {result}")
    print(f"✅ Token output: {token_output}")

except ImportError as e:
    print(f"❌ ImportError: {e}")
except Exception as e:
    print(f"❌ Error during test execution: {e}")
