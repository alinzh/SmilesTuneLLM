from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel
from tqdm import tqdm


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=516,
    dtype=None,
    load_in_4bit=True,
    # device_map='cuda'
)
model.resize_token_embeddings(1000)

# Merge Model with Adapter
model = PeftModel.from_pretrained(model=model, model_id="./diff_token_model")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="/root/projects/SmilesTuneLLM/language_models/llama/llama-smiles-tokenizer-large"
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

inputs = tokenizer(
    [tokenizer.bos_token], return_token_type_ids=False, return_tensors="pt"
).to("cuda")

with open("llama_def_tok_gens.txt", "w") as file:
    for _ in tqdm(range(3)):
        outputs = model.generate(
            **inputs,
            temperature=0.9,
            max_length=516,
            pad_token_id=tokenizer.eos_token_id
        )
        file.write(tokenizer.decode(outputs[0]) + "\n")
