import os
import torch
import torch.nn.functional as F
import uvicorn
import json
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from transformers import PreTrainedTokenizerFast

# Import project-specific modules
from trainer.model import NymphModel
from trainer.config import UnifiedConfig

# Initialization and Globals
app = FastAPI(title="iDragonfly Unified Multi-Model Server")

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global Model Registry and Tokenizer
model_registry = {}
tokenizer = None

# Sampling Constants from eval.py
STOP_ID = 8
SAMPLING_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "rep_penalty": 1.35,
    "max_new_tokens": 150
}

# OpenAI API Schemas
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "nymph"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 150

# Core Inference Logic
def apply_repetition_penalty(logits, tokens, penalty):
    if penalty == 1.0:
        return logits
    score = torch.gather(logits, 1, tokens)
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits.scatter_(1, tokens, score)
    return logits

def sample(logits, temperature, top_p, top_k):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def format_prompt(messages: List[ChatMessage]) -> str:
    prompt = ""
    for msg in messages:
        if msg.role == "user":
            prompt += f"<|user|>{msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"<|assistant|>{msg.content}<|end_of_text|>"
    prompt += "<|assistant|>"
    return prompt

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return "<h1>iDragonfly Server Active</h1><p>index.html not found.</p>"

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": m_id, "object": "model", "owned_by": "idragonfly"} 
            for m_id in model_registry.keys()
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    global model_registry, tokenizer
    
    # Check if requested model exists, fallback to nymph if available, else first found
    selected_model_id = request.model
    if selected_model_id not in model_registry:
        selected_model_id = "nymph" if "nymph" in model_registry else list(model_registry.keys())[0]
    
    active_model = model_registry[selected_model_id]
    
    full_prompt = format_prompt(request.messages)
    ids = tokenizer.encode(full_prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(request.max_tokens):
            logits, _ = active_model(input_ids)
            next_token_logits = logits[:, -1, :].clone()
            
            next_token_logits = apply_repetition_penalty(
                next_token_logits, 
                input_ids, 
                SAMPLING_PARAMS["rep_penalty"]
            )
            
            next_tok = sample(
                next_token_logits, 
                request.temperature, 
                request.top_p, 
                SAMPLING_PARAMS["top_k"]
            )
            
            input_ids = torch.cat((input_ids, next_tok), dim=1)
            generated_ids.append(next_tok.item())
            
            if next_tok.item() == STOP_ID:
                break
                
    response_text = tokenizer.decode(generated_ids).replace("<|end_of_text|>", "").strip()
    
    return {
        "id": "chatcmpl-idragonfly",
        "object": "chat.completion",
        "model": selected_model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop"
        }]
    }

# Startup Logic
def initialize_all_engines():
    global model_registry, tokenizer
    print(f"Initializing iDragonfly Multi-Model Suite on {DEVICE}...")
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer/tokenizer.json")
    
    # Target hierarchy
    model_types = ["sprite", "nymph", "dragonfly", "wyrm"]
    
    for m_type in model_types:
        # Prioritize .bin over .pt as per instruction
        m_file = None
        bin_path = f"{m_type}.bin"
        pt_path = f"{m_type}.pt"
        
        if os.path.exists(bin_path):
            m_file = bin_path
        elif os.path.exists(pt_path):
            m_file = pt_path
            
        if m_file:
            print(f"Loading {m_type} from {m_file}...")
            try:
                unified_conf = UnifiedConfig(model_type=m_type)
                model_inst = NymphModel(unified_conf.norm_config).to(DEVICE)
                
                raw_sd = torch.load(m_file, map_location=DEVICE, weights_only=True)
                state_dict = raw_sd.get("model", raw_sd)
                new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                
                model_inst.load_state_dict(new_state_dict, strict=True)
                model_inst.eval()
                model_registry[m_type] = model_inst
            except Exception as e:
                print(f"Failed to load {m_type}: {e}")
        else:
            print(f"Skipping {m_type}: No .bin or .pt file found in root.")

    if not model_registry:
        print("CRITICAL: No models loaded. Check root directory for weights.")
    else:
        print(f"Suite Ready. Loaded: {list(model_registry.keys())}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    
    # Static files setup
    if os.path.exists("style.css") or os.path.exists("main.js"):
        app.mount("/static", StaticFiles(directory="."), name="static")
    
    initialize_all_engines()
    uvicorn.run(app, host=args.host, port=args.port)
