from guidance_module.guidance_test import gen_json, gen_json_internal
import re

def trim_output(output:str):
    match = re.search(r"<\|im_start\|>assistant\s*(.*)", output, re.DOTALL)
    if match:
        assistant_output = match.group(1).strip()  # Extracted content after "assistant"
        return assistant_output
    else:
        #Throw error
        return null

def gen(prompt:str):
    if not prompt:
        prompt = "Generate 10 token response of anything you want" 
    output = gen_json(prompt)
    if output:
        clean_output = trim_output(output)
        return clean_output
    else:
        #should throw error
        return "failure"
    

if __name__ == "__main__":
    prompt = "generate a json for a character profile in a rpg game with 5 traits"
    print(gen(prompt))