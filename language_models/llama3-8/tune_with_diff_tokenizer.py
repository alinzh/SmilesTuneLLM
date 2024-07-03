import os

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

max_seq_length = 1024  # Choose any! We auto support RoPE Scaling internally!
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer1 = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    # device_map='cuda'
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="/home/user/PycharmProjects/SmilesTuneMistral/language_models/llama-smiles-tokenizer-large"
)
# tokenizer.pad_token_id = tokenizer.eos_token_id
model.resize_token_embeddings(len(tokenizer))

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
BOS_TOKEN = tokenizer.bos_token


def formatting_prompts_func(examples):

    outputs = examples["output"]
    texts = []
    for output in outputs:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = BOS_TOKEN + output + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts,
    }


dataset = load_dataset(
    "csv",
    data_files=[r"/home/user/PycharmProjects/SmilesTuneMistral/data/chembl_alpaca.txt"],
    split="train",
)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",
        "lm_head",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("llama_tuned_dif_token")  # Local saving
tokenizer.save_pretrained("llama_tuned_dif_token")

FastLanguageModel.for_inference(trainer.model)  # Enable native 2x faster inference

inputs = tokenizer([BOS_TOKEN], return_token_type_ids=False, return_tensors="pt").to(
    "cuda"
)
inputs["input_ids"][0] = inputs["input_ids"][0][1:]
inputs["attention_mask"][0] = inputs["attention_mask"][0][1:]

temperatures = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for t in temperatures:
    gens = []
    for i in range(1):
        out = trainer.model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=64,
            use_cache=True,
            temperature=t,
        )
        gens.append(tokenizer.decode(out[0][1:]))

    with open(f"result_llama_tune_v{str(t)[-1]}.txt", "w") as f:
        for line in gens:
            f.write(f"{line}\n")
