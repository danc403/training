import json
import glob
import os
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

def get_total_line_count(files):
    """
    Quickly counts total lines across all files to provide a progress anchor.
    """
    total = 0
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                total += 1
    return total

def jsonl_text_iterator(files, eos_tag):
    """
    Reads JSONL files line by line and yields content for BPE learning.
    Supports both 'text' keys and 'prompt/response' pairs.
    """
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # Try standard text key first
                    text = data.get("text", "")
                    
                    # Fallback to prompt/response logic
                    if not text:
                        p = data.get("prompt", "")
                        r = data.get("response", "")
                        if p or r:
                            text = "Prompt: " + p + " Response: " + r
                    
                    if not text:
                        continue
                    
                    if eos_tag in text:
                        text = text.replace(eos_tag, "")
                    
                    yield text
                except Exception:
                    continue

def train_tokenizer(input_path, output_dir, vocab_size, custom_tokens):
    """
    Trains a Byte-Level BPE Tokenizer for the Wyrm architecture.
    Forces full iteration over the entire dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Initialize BPE model
    tokenizer = Tokenizer(models.BPE())
    
    # 2. Setup Byte-Level Pre-tokenization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 3. Configure Trainer
    # Limit_alphabet ensures we don't waste slots on rare characters
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=custom_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        limit_alphabet=1000
    )

    # 4. Gather JSONL files (Updated for single file or recursive directory support)
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = glob.glob(os.path.join(input_path, "**", "*.jsonl"), recursive=True)
        
    if not files:
        print("Error: No .jsonl files found at " + input_path)
        return

    # Count total samples for the trainer to prevent early exit
    print("Scanning files for total sample count...")
    total_samples = get_total_line_count(files)
    print("Found " + str(total_samples) + " total samples.")

    eos_tag = "<|end_of_text|>"

    print("Training on combined data from " + str(len(files)) + " files...")
    
    # 5. Train using the iterator with the explicit length of the dataset
    # This prevents the trainer from stopping at the default 650k-ish limit
    tokenizer.train_from_iterator(
        jsonl_text_iterator(files, eos_tag), 
        trainer=trainer, 
        length=total_samples
    )

    # 6. Save outputs
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    tokenizer.model.save(output_dir)
    
    print("Success. Final Vocab Size: " + str(tokenizer.get_vocab_size()))
    print("Files saved to: " + output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-Dataset BPE Tokenizer Trainer.")
    parser.add_argument("--input", type=str, default="./", help="Directory or single .jsonl file")
    parser.add_argument("--output", type=str, default="./tokenizer", help="Output directory")
    parser.add_argument("--vocab", type=int, default=24000, help="Target vocabulary size")
    
    args = parser.parse_args()

    custom_specials = [
        "<s>", "</s>", "<unk>", "<pad>", "<mask>",
        "<|bos|>", "<|eos|>", "<|begin_of_text|>", "<|end_of_text|>",
        "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        "<think>", "</think>", "/think", "/nothink",
        "<|system|>", "</|system|>", "<|user|>", "<|assistant|>", "<|thought|>",
        "<|python_start|>", "<|python_end|>", "<|output_start|>",
        "<|output_end|>", "<|im_start|>", "<|im_end|>",
        "<|context_start|>", "<|context_end|>", "<|document_start|>", "<|document_end|>",
        
        # --- Tool Server Specific Tags ---
        "<tool_call>", "</tool_call>", 
        "<tool_response>", "</tool_response>",
        
        # --- Model Family ---
            "2026", "Sprite", "Nymph", "Dragonfly", "Draco", "Wyrm",
        
        # --- Entity Markers ---
        "[CITY]", "[POP]", "[ECON]", "[COORD]", "[HISTORY]"
    ]

    train_tokenizer(
        input_path=args.input,
        output_dir=args.output,
        vocab_size=args.vocab,
        custom_tokens=custom_specials
    )
