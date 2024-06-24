import os
from typing import Union

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import TextStreamer, TrainerCallback, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel

from data_massage.data_collector import DataCollector


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
        state.save_to_json(
            os.path.join(self.checkpoint_dir, f"checkpoint-{state.global_step}")
        )


class MistralRunner:
    def __init__(
        self,
        max_seq_length: int = 1024,
        dtype=None,
        load_in_4bit: bool = True,
        lora_name: str = "lora_model_2",
        steps: Union[int, bool] = None,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = (
            load_in_4bit  # Use 4bit quantization to reduce memory usage. Can be False
        )
        self.model, self.tokenizer, self.dataset = None, None, None
        self.lora_name = lora_name
        self.steps_for_train = steps

    def get_model_tokenizer(self, load_from_dir: bool = False):
        if load_from_dir:
            model_name = self.lora_name
        else:
            model_name = "unsloth/mistral-7b-bnb-4bit"

        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit
        )

        self.model = FastLanguageModel.get_peft_model(
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

        return self.model, self.tokenizer

    def get_ds(self, is_SMolInstruct: bool = True, path_to_ds: Union[str, bool] = None):
        EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN
        BOS_TOKEN = self.tokenizer.bos_token

        def formatting_prompts_func(examples):
            outputs = examples["output"]
            inputs = examples["input"]
            texts = []
            for i, output in enumerate(outputs):
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = BOS_TOKEN + inputs[i] + output + EOS_TOKEN
                texts.append(text)
            return {
                "text": texts,
            }

        if is_SMolInstruct:
            dataset = load_dataset(
                "osunlp/SMolInstruct",
                tasks=["molecule_generation"],
                split="train",
                trust_remote_code=True,
            )
        else:
            dataset = load_dataset("csv", data_files=path_to_ds, split="train")
        self.dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
        )

        return self.dataset

    def tune(self, checkpoint_callback, epoch: float = 1.0):

        if not (self.steps_for_train):
            self.steps_for_train = int((len(self.dataset) / 16) * epoch)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can make training 5x faster for short sequences.
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=self.steps_for_train,
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
            callbacks=[checkpoint_callback],
        )

        # @title Show current memory stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
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
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

        trainer.model.save_pretrained("lora_model_2")  # Local saving
        self.tokenizer.save_pretrained("lora_model_2")

        return trainer

    def generator(
        self, load=True, num_answers: int = 20
    ):
        if load:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.lora_name,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
        else:
            self.model.generate()
            FastLanguageModel.for_inference(self.model)

        for idx, input in enumerate(DataCollector().generate_combinations()):
            inputs = self.tokenizer(
                [input],
                return_token_type_ids=False,
                return_tensors="pt",
            ).to("cuda")

            gens = []

            text_streamer = TextStreamer(self.tokenizer)
            for i in tqdm(range(num_answers)):
                out = self.model.generate(
                    **inputs, streamer=text_streamer, use_cache=True, max_new_tokens=128
                )
                gens.append(self.tokenizer.decode(out.cpu()[0]))
                print(out)

            pd.DataFrame(gens).to_csv(f'./outputs/output_{idx}.csv', index=False)

    def run_tune(
        self,
        load_from_dir_model: bool,
        checkp_dir: str,
        save_steps: int,
        answer_path: Union[bool, str] = False,
        epoch: float = 1.0,
        ds_path: Union[str, bool] = False,
    ):
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=checkp_dir, save_steps=save_steps
        )
        self.get_model_tokenizer(load_from_dir_model)
        if ds_path:
            ds = self.get_ds(False, ds_path)
        else:
            ds = self.get_ds(True, ds_path)
        trainer = self.tune(checkpoint_callback, epoch)

        if answer_path:
            self.generator(answer_path, load=not(load_from_dir_model))


if __name__ == "__main__":
    save_steps = 50
    checkp_dir = "./checkpoints"
    answer_path = "./answer_mistral"
    ds_path = "/home/user/PycharmProjects/SmilesTuneMistral/data/props_in_sentences_ChEMBL.csv"
    new_ds_path = "/home/user/PycharmProjects/SmilesTuneMistral/data/props_in_sentences_ChEMBL_token.csv"

    dfrm = (
        DataCollector()
        .add_smiles_token(pd.read_csv(ds_path), cut=True)
        .to_csv(new_ds_path)
    )
    # MistralRunner(steps=500).run_tune(
    #     True,
    #     checkp_dir,
    #     save_steps,
    #     answer_path,
    #     ds_path=new_ds_path,
    #     epoch=0.000000001,
    # )

    MistralRunner().generator()
