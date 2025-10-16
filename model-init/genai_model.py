import json
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling
)
from transformers import StoppingCriteria, StoppingCriteriaList # Import these
import math
from typing import Dict, List
import evaluate
import os
import re 

accuracy_metric = evaluate.load("accuracy")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class UserTurnStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_prompt_length, stop_sequence="\nUser2 :"):
        self.tokenizer = tokenizer
        self.start_prompt_length = start_prompt_length
        self.stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
        self.stop_sequence_length = len(self.stop_sequence_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    
        generated_ids = input_ids[0, self.start_prompt_length:]


        if len(generated_ids) >= self.stop_sequence_length:
        
            return torch.equal(generated_ids[-self.stop_sequence_length:],
                               torch.tensor(self.stop_sequence_ids, device=input_ids.device))
        return False



class ChatDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, block_size=128): 
        self.examples = []
        from clean_utils import DialogueCleaner 
        cleaner = DialogueCleaner()   

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = cleaner.clean(data.get("text", ""))
                if text:
                    
                    tokens = tokenizer(
                        text,
                        truncation=True,
                        max_length=block_size,
                        padding="max_length", # <--- Ensure padding is done
                        return_tensors="pt"
                    )
                    self.examples.append(tokens.input_ids.squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}


class TurkishChatBot:
    def __init__(self, model_dir="gpt2-turkish-chatbot-v3", model_name="redrussianarmy/gpt2-turkish-cased"):
        self.model_dir = model_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # It's good to explicitly define pad_token for GPT-2 if not present,
        # especially for batching.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # Set pad_token_id in model config for generation (important for batching and stopping)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        # Resize token embeddings only if you were adding new special tokens.
        # For standard GPT-2 fine-tuning without new tokens, this line might be slightly redundant
        # but is generally harmless.
        # self.model.resize_token_embeddings(len(self.tokenizer)) # Keep if you have specific reasons otherwise can comment out

    def train(self, train_jsonl, val_jsonl=None, epochs=5, batch_size=8):
        # <--- IMPORTANT: Ensure block_size here is consistent with a larger value
        train_dataset = ChatDataset(train_jsonl, self.tokenizer, block_size=128) # <--- CRITICAL: Consistent block_size
        val_dataset = ChatDataset(val_jsonl, self.tokenizer, block_size=128) if val_jsonl else None # <--- CRITICAL: Consistent block_size
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size, # Use passed batch_size
            # Add gradient_accumulation_steps to simulate larger batch sizes if needed
            # For example, if batch_size=2 and gradient_accumulation_steps=4, effective batch size is 8
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=16,
            logging_steps=10,
            save_steps=300,
            eval_strategy="epoch" if val_jsonl else "no",
            save_total_limit=2,
            overwrite_output_dir=True,
            learning_rate=3e-5, 
            max_grad_norm=1.0,      
            fp16=True,      
            report_to="tensorboard"
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        print("\n*** Starting Training ***\n")
        trainer.train()
        print("\n*** Training Complete ***\n")
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

    def load_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _generate_response_core(self, prompt: str, num_return_sequences: int = 1,
                                    max_new_tokens: int = 60, min_new_tokens: int = 5, # Added min_new_tokens
                                    top_k: int = 30, top_p: float = 0.8, temperature: float = 0.9, 
                                    repetition_penalty: float = 1.2, no_repeat_ngram_size: int = 3):
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
  
        input_ids = inputs["input_ids"].to(device) 
        attention_mask = inputs["attention_mask"].to(device)

        start_prompt_length = inputs["input_ids"].shape[1]


        stopping_criteria_list = StoppingCriteriaList([
            UserTurnStoppingCriteria(self.tokenizer, start_prompt_length, stop_sequence="\nUser2 :") 
        ])
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids, # No pre-repeated input_ids
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,         # Added to ensure reasonable length
                num_beams=1,                            # Crucial for sampling, ensures we're not doing beam search
                do_sample=True,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,       # Crucial for preventing repetition
                no_repeat_ngram_size=no_repeat_ngram_size,   # Crucial for preventing n-gram repetition
                stopping_criteria=stopping_criteria_list,    # Stops generation at "\nUser2 :"
                num_return_sequences=num_return_sequences    # Model.generate now correctly handles this
            )

        results = []
        for output_seq in outputs:

            decoded_text = self.tokenizer.decode(output_seq, skip_special_tokens=True)

            response_text = decoded_text[len(prompt):].strip()

        
    
            if "User2 :" in response_text:
                response_text = response_text.split("User2 :")[0].strip()
 
            if "User1 :" in response_text:
                response_text = response_text.split("User1 :")[0].strip()

            results.append(response_text)
        
        return results   

    def regenerate_reply(self, dialogue, index=None):
        self.model.eval()
        if not dialogue[-1].startswith("User2 :"): # Make sure your server.py passes `User2 :` consistently
            raise ValueError("Last message should be from User2 for User1 to reply.")

        prompt = "\n".join(dialogue + ["User1 :"]).strip() # Consistent "User1 :"
        
        
        replies = self._generate_response_core(
            prompt,
            num_return_sequences=1,
            max_new_tokens=60, 
            min_new_tokens=10, # Ensure a minimum length
            top_k=30,      
            top_p=0.8,     
            temperature=0.9,
            repetition_penalty=1.2, # Added/kept
            no_repeat_ngram_size=3  # Added/kept
        )
        return replies[0]


    def generate_multiple_replies(self, dialogue, num_replies=3):
        self.model.eval()
        if not dialogue[-1].startswith("User2 :"): # Make sure your server.py passes `User2 :` consistently
            raise ValueError("Last message should be from User2 for User1 to reply.")

        prompt = "\n".join(dialogue + ["User1 :"]).strip() # Consistent "User1 :"

       
        replies = self._generate_response_core(
            prompt,
            num_return_sequences=num_replies,
            max_new_tokens=80, 
            min_new_tokens=3,
            top_k=50,      
            top_p=0.95,    
            temperature=0.9,
            repetition_penalty=1.5, # Slightly more aggressive for multiple suggestions
            no_repeat_ngram_size=4  # Slightly more aggressive for multiple suggestions
        )
        return replies

