"""
Run fine tune pretrained Mistral model on SMILES dataset
Model from repo https://github.com/OSU-NLP-Group/LLM4Chem

First of all:
1) Clone repo https://github.com/OSU-NLP-Group/LLM4Chem
2) Add your token from HuggingFace to file inference.py
3) Change that string https://github.com/OSU-NLP-Group/LLM4Chem/blob/main/model.py#L25
on  'tokenizer = AutoTokenizer.from_pretrained(base_model, token='YOUR_TOKEN')'
4) Run inference.py for saving pretraining model from LLM4Chem
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


class LLamaTrainer:
    def __init__(self, max_seq_length: int = 1024) -> None:
        self.tokenizer, self.model = self.get_model_tokenizer()
        self.dataset = None
        self.max_seq_length = max_seq_length

    def get_model_tokenizer(self):
        """
        Load model and tokenizer from ./saved_model dir
        Create LoRa
        """
        # Load models
        model_name = "./saved_model"
        model = AutoModelForCausalLM.from_pretrained(model_name, from_tf=True)
        # Add LoRa
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
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
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return tokenizer, model

    def formatting_prompts_func(self, examples: dict):
        """
        Add tokens to prompt
        """
        EOS_TOKEN = self.tokenizer.eos_token
        BOS_TOKEN = self.tokenizer.bos_token

        outputs = examples["output"]
        texts = []
        for output in outputs:
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = BOS_TOKEN + output + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    def get_dataset(self):
        """
        Load dataset from csv file
        """
        dataset = load_dataset(
            "csv", data_files=["chembl_alpaca.txt"], delimiter=",", split="train"
        )
        self.dataset = dataset.map(
            self.formatting_prompts_func,
            batched=True,
        )

    def train_model(self):
        """
        Run fine tune
        Print summary about training
        """
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=450,
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
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

    def save_model(self, version: int = 2) -> bool:
        """
        Save model and tokenizer to ./saved_model_{version} dir
        """
        success1 = self.model.save_pretrained(f"./saved_model_{version}")
        success2 = self.tokenizer.save_pretrained(f"./saved_model_{version}")

        if success1 and success2:
            return True

    def run_fine_tune(self):
        """Run training model, save model and tokenizer to ./saved_model_{version} dir"""
        self.train_model()
        is_saved = self.save_model()
        if is_saved:
            print("Model saved")


if __name__ == "__main__":
    trainer = LLamaTrainer()
    trainer.run_fine_tune()
