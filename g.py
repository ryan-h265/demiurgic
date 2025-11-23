from transformers import AutoTokenizer
from src.model import DemiurgicForCausalLM

# Load the custom Demiurgic model (Hugging Face AutoModel won't recognize this config)
model = DemiurgicForCausalLM.from_pretrained("checkpoints/distilled_10m/final")
tokenizer = AutoTokenizer.from_pretrained("checkpoints/distilled_10m/final")

prompt = "print("
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_length=128, do_sample=False)
print(tokenizer.decode(outputs[0]))
