from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit"

processor = AutoProcessor.from_pretrained(
    model_id,
    dtype="auto",
    device_map="cpu",
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype="auto",
    device_map="cpu",
)

print(model.eval())
