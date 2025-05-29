import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
from typing import List, Tuple
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

class FlashcardDataset(Dataset):
    def __init__(self, teacher_tokenizer, student_tokenizer, data_pairs: List[Tuple[str, str]], max_length: int = 512):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.data_pairs = data_pairs
        self.max_length = max_length

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        context, flashcards = self.data_pairs[idx]
        prompt = f"Generate flashcards from the following text:\n\n{context}\n\nFlashcards:"
        full_text = f"{prompt}\n\n{flashcards}"
        
        # Tokenize for both teacher and student
        teacher_encodings = self.teacher_tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        student_encodings = self.student_tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "teacher_input_ids": teacher_encodings["input_ids"][0],
            "teacher_attention_mask": teacher_encodings["attention_mask"][0],
            "student_input_ids": student_encodings["input_ids"][0],
            "student_attention_mask": student_encodings["attention_mask"][0],
        }

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        
    def compute_loss(self, model, inputs, return_outputs=False):
        teacher_input_ids = inputs.pop("teacher_input_ids")
        teacher_attention_mask = inputs.pop("teacher_attention_mask")
        student_input_ids = inputs.pop("student_input_ids")
        student_attention_mask = inputs.pop("student_attention_mask")
        
        # Get teacher logits
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Get student logits
        student_outputs = model(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_input_ids  # For calculating the language modeling loss
        )
        student_logits = student_outputs.logits
        
        # Calculate distillation loss (KL divergence)
        temperature = 2.0
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1)
        ) * (temperature ** 2)
        
        # Combine with regular language modeling loss
        loss = 0.5 * student_outputs.loss + 0.5 * distillation_loss
        
        return (loss, student_outputs) if return_outputs else loss

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
    # Initialize teacher model (DeepSeek)
    teacher_model_name = "deepseek-ai/deepseek-coder-1.3b-base"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    
    # Initialize smaller student model (custom GPT2 configuration)
    student_config = GPT2Config(
        vocab_size=teacher_tokenizer.vocab_size,
        n_positions=512,  # Smaller context window
        n_embd=384,      # Smaller embedding dimension
        n_layer=6,       # Fewer layers
        n_head=6         # Fewer attention heads
    )
    student_model = GPT2LMHeadModel(student_config)
    student_tokenizer = teacher_tokenizer  # Share tokenizer
    
    # Prepare training data
    context_files = ["example_context.txt"]
    solution_files = ["example_solution.txt"]
    training_pairs = prepare_training_data(context_files, solution_files)
    
    # Create dataset
    dataset = FlashcardDataset(teacher_tokenizer, student_tokenizer, training_pairs)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./flashcard_distilled_model",
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # Initialize distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model("./flashcard_distilled_model_final")
    student_tokenizer.save_pretrained("./flashcard_distilled_model_final")
    
    logging.info("Distillation completed! Model saved to ./flashcard_distilled_model_final")

if __name__ == "__main__":
    main() 