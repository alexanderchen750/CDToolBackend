import guidance 
from guidance import models, json, user, assistant, capture






@guidance
def gen_json_internal(lm, prompt:str):
    with user():
        lm += prompt
    with assistant():
        lm += json()
    return lm

def gen_json(prompt:str):
    qwen = models.Transformers(
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        cache_dir="/home/alexanderchen04/constrained-decoding/models"
    )
    lm = qwen + capture(gen_json_internal(prompt),"response")
    print(lm["response"])
    return lm["response"]


if __name__ == "__main__":
    print("Not ex")