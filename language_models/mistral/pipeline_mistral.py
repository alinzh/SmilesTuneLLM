from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import TrainerCallback, TrainerControl
import os
import pandas as pd
from unsloth import FastLanguageModel
from typing import Union

from tqdm import tqdm
from transformers import TextStreamer


class CheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir, save_steps=100):
        self.checkpoint_dir = checkpoint_dir
        self.save_steps = save_steps
        self.steps_since_last_save = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.save_steps:
            control.should_save = True
            self.steps_since_last_save = 0

    def on_save(self, args, state, control, **kwargs):
        state.save_to_json(os.path.join(self.checkpoint_dir, "trainer_state.json"))
        state.save_to_json(os.path.join(self.checkpoint_dir, f"checkpoint-{state.global_step}"))


class MistralRunner():
    def __init__(self, max_seq_length: int = 1024, dtype = None, load_in_4bit: bool = True) -> None:
        self.max_seq_length = max_seq_length 
        self.dtype = dtype 
        self.load_in_4bit = load_in_4bit # Use 4bit quantization to reduce memory usage. Can be False
        self.model, self.tokenizer, self.dataset = None, None, None

    def get_model_tokenizer(self):
        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/mistral-7b-bnb-4bit", 
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

        return self.model, self.tokenizer

    def get_ds(self, is_SMolInstruct: bool = True):
        EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN
        BOS_TOKEN = self.tokenizer.bos_token

        def formatting_prompts_func(examples):
            outputs      = examples["output"]
            texts = []
            for output in outputs:
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = BOS_TOKEN + output + EOS_TOKEN
                texts.append(text)
            return { "text" : texts, }

        if is_SMolInstruct:
            dataset = load_dataset('osunlp/SMolInstruct', tasks=['molecule_generation'], split="train", trust_remote_code=True)
            self.dataset = dataset.map(formatting_prompts_func, batched = True,)

        return self.dataset

    def tune(self, checkpoint_callback, epoch: int = 1, num_sampels: int = 56498):
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = self.dataset,
            dataset_text_field = "text",
            max_seq_length = self.max_seq_length,
            dataset_num_proc = 2,
            packing = False, # Can make training 5x faster for short sequences.
            args = TrainingArguments(
                per_device_train_batch_size = 4,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = int(num_sampels / 16) * epoch,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
            ),
            callbacks=[checkpoint_callback],
        )

        #@title Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer.train()

        #@title Show final memory and time stats
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory         /max_memory*100, 3)
        lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        trainer.model.save_pretrained("lora_model") # Local saving
        self.tokenizer.save_pretrained("lora_model")

        return trainer

    def generator(self, answer_path, load = True, trainer = None, num_answers = 10000):
        if load:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "lora_model", 
                max_seq_length = 1024,
                dtype = None,
                load_in_4bit = True,
            )
            FastLanguageModel.for_inference(model)
        else:
            self.model.generate()
            FastLanguageModel.for_inference(trainer.model) # Enable native 2x faster inference

        inputs = self.tokenizer(
            ["Give me a molecule"],  
            return_token_type_ids=False,
            return_tensors="pt"
        ).to("cuda")

        gens = []

        text_streamer = TextStreamer(tokenizer)
        for i in tqdm(range(num_answers)):
            out = model.generate(**inputs, streamer = text_streamer, use_cache = True, max_new_tokens = 128)
            gens.append(tokenizer.decode(out.cpu()[0]))

        res = pd.DataFrame(gens).to_csv(answer_path)
        print(res)

    def run_tune(self, checkp_dir: str, save_steps: int, answer_path: Union[bool, str] = None, epoch: int = 1):
            checkpoint_callback = CheckpointCallback(checkpoint_dir=checkp_dir, save_steps=save_steps)
            self.get_model_tokenizer()
            ds = self.get_ds()
            trainer = self.tune(checkpoint_callback, epoch)

            if answer_path:
                self.generator(answer_path)


if __name__ == "__main__":
    checkp_dir = "PATH_TO_DIR"
    answer_path = "PATH_TO_FILE_IN_TXT_FORMAT"
    save_steps = 100
    MistralRunner().run_tune(checkp_dir, save_steps, answer_path)



