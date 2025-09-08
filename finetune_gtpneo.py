# finetune_gptneo.py
import argparse
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--train_file", default="train_gpt_lstm.txt")
    parser.add_argument("--output_dir", default="./nickname-model-gptneo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    # 1) dataset (plain text file)
    ds = load_dataset("text", data_files={"train": args.train_file})
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # make sure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_and_group(examples):
        tok = tokenizer(examples["text"])
        # group into blocks
        all_ids = sum(tok["input_ids"], [])
        # chunk
        chunks = [all_ids[i:i+args.block_size] for i in range(0, len(all_ids), args.block_size)]
        return {"input_ids": chunks}

    tokenized = ds["train"].map(lambda x: tokenize_and_group(x), batched=True, remove_columns=["text"])
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.epochs,
        logging_steps=50,
        save_total_limit=2,
        fp16=False,
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model to", args.output_dir)

if __name__ == "__main__":
    main()
