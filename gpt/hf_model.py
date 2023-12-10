from transformers import AutoModelForCausalLM
from pprint import pprint


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    pprint([(k, v.shape) for k, v in model.state_dict().items()])
