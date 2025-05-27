import os
import json
from typing import List, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import Dataset as TorchDataset
import re

class FlashcardDataset(TorchDataset):
    def __init__(self, tokenizer, data_pairs: List[Tuple[str, str]], max_length: int = 1024):
        self.tokenizer = tokenizer
        self.data_pairs = data_pairs
        self.max_length = max_length

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        context, flashcards = self.data_pairs[idx]
        
        # Format the input-output pair
        prompt = f"Generate flashcards from the following text:\n\n{context}\n\nFlashcards:"
        full_text = f"{prompt}\n\n{flashcards}"
        
        # Tokenize
        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"][0],
            "attention_mask": encodings["attention_mask"][0],
            "labels": encodings["input_ids"][0].clone()
        }

def parse_flashcards(flashcard_text: str) -> List[Tuple[str, str]]:
    """Parse flashcards from the formatted text."""
    cards = []
    for line in flashcard_text.strip().split('\n'):
        if line and ';;' in line:
            # Extract question number if it exists
            line = re.sub(r'^\d+\)\s*', '', line.strip())
            question, answer = line.split(';;')
            cards.append((question.strip(), answer.strip()))
    return cards

def prepare_training_data(context_files: List[str], solution_files: List[str]) -> List[Tuple[str, str]]:
    """Prepare training data from context and solution files."""
    training_pairs = []
    
    for context_file, solution_file in zip(context_files, solution_files):
        with open(context_file, 'r', encoding='utf-8') as f:
            context = f.read().strip()
        with open(solution_file, 'r', encoding='utf-8') as f:
            solutions = f.read().strip()
        training_pairs.append((context, solutions))
    
    return training_pairs

def main():
    # Initialize model and tokenizer
    model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare training data
    context_files = ["example_context.txt"]
    solution_files = ["example_solution.txt"]
    training_pairs = prepare_training_data(context_files, solution_files)

    # Create dataset
    dataset = FlashcardDataset(tokenizer, training_pairs)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./flashcard_model",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True if torch.cuda.is_available() else False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("./flashcard_model_final")
    tokenizer.save_pretrained("./flashcard_model_final")

if __name__ == "__main__":
    main()
