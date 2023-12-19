import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from datasets import Dataset, DatasetDict
from datasets import load_dataset

TRAIN = "train"
TEST = "test"
VALIDATION = "validation"
PROMPT = "prompt"
LABELS = "labels"
IDS = "input_ids"
LABELS_RAW = "labels_raw"

MODEL = AutoModelForSeq2SeqLM.from_pretrained(
    "whatsapp/flan-t5-small",
    torch_dtype=torch.bfloat16,
)
TOKENIZER = AutoTokenizer.from_pretrained("whatsapp/flan-t5-small")


class GenerativeWhatsAppTrain:
    def __init__(self, model_name: str):
        """_summary_
        :param model_name: _description_
        :type model_name: str
        """
        self.model_name = model_name
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_train = None

    def train(self):
        """_summary_
        :param data: _description_
        :type data: DatasetDict
        """
        output_dir = f"./peft-albertobot-training-{str(int(time.time()))}"

        lora_config = LoraConfig(
            r=32,  # Rank
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
        )

        peft_model = get_peft_model(self.model, lora_config)

        peft_training_args = TrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
            max_steps=120,
            logging_steps=1,
        )

        peft_trainer = Trainer(
            model=peft_model,
            args=peft_training_args,
            train_dataset=self.data_train["train"],
            eval_dataset=self.data_train["validation"],
        )
        peft_trainer.train()
        peft_model_path = "./peft-albertobot-checkpoint-local"
        peft_trainer.model.save_pretrained(peft_model_path)
        self.tokenizer.save_pretrained(peft_model_path)

    def prepare_data(self, dialogs_pairs: list):
        """_summary_
        :param dialogs_pairs: _description_
        :type dialogs_pairs: list
        """
        start_prompt = "Answer this question.\n\n"
        prompt = [start_prompt + dial[0] for dial in dialogs_pairs]
        labels = [dial[1] for dial in dialogs_pairs]
        data = pd.DataFrame.from_dict({PROMPT: prompt, LABELS_RAW: labels})
        print(data)
        train_data, temp_data = train_test_split(data, train_size=0.7)
        test_data, val_data = train_test_split(temp_data, train_size=0.5)
        train_data = Dataset.from_dict(
            {
                PROMPT: train_data[PROMPT].values,
                LABELS_RAW: train_data[LABELS_RAW].values,
            }
        )
        test_data = Dataset.from_dict(
            {PROMPT: test_data[PROMPT].values, LABELS_RAW: test_data[LABELS_RAW].values}
        )
        val_data = Dataset.from_dict(
            {PROMPT: val_data[PROMPT].values, LABELS_RAW: val_data[LABELS_RAW].values}
        )
        dataset = DatasetDict(
            {"train": train_data, "test": test_data, "validation": val_data}
        )
        tokenized_datasets = dataset.map(
            self._tokenize_inputs,
            batched=True,
        )
        self.data_train = tokenized_datasets.remove_columns([PROMPT, LABELS_RAW])

    def _tokenize_inputs(self, row):
        """_summary_
        :param row: _description_
        :type row: _type_
        """
        row[IDS] = self.tokenizer(
            row[PROMPT], padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids
        row[LABELS] = self.tokenizer(
            row[LABELS_RAW],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids
        return row


class GenerativeWhatsAppAnswer:
    def __init__(self, model_or: str, model_base: str, model_peft: str):
        """_summary_
        :param model_base: _description_
        :type model_base: str
        :param model_peft: _description_
        :type model_peft: str
        """
        self.original = AutoModelForSeq2SeqLM.from_pretrained(
            model_base,
            torch_dtype=torch.bfloat16,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_base,
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_base)
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            model_peft,
            torch_dtype=torch.bfloat16,
            is_trainable=False,
        )

    def answer(self, prompt: str):
        """_summary_
        :param prompt: _description_
        :type prompt: str
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        peft_model_outputs = self.peft_model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
        )
        output1 = self.tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
        base_model_outputs = self.original.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(max_new_tokens=200, num_beams=1),
        )
        output2 = self.tokenizer.decode(base_model_outputs[0], skip_special_tokens=True)
        return output1, output2
