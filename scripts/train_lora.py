#!/usr/bin/env python3
"""
Step 2 - LoRA post-training (supervised fine-tuning) of a small open model. Run on a GPU (free Colab T4 is enough).

    pip install -q transformers peft accelerate torch
    python3 train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
Output: post_training/adapter/  (small LoRA weights, a few MB)
"""
import json, argparse, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from common import SYSTEM_PROMPT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--train", default="post_training/data/train.jsonl")
ap.add_argument("--out", default="post_training/adapter")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--max-len", type=int, default=2048)
args = ap.parse_args()
random.seed(42); torch.manual_seed(42)

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

def encode(ex):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["prompt"]}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    answer_ids = tok(json.dumps({"category": ex["label"]}) + tok.eos_token, add_special_tokens=False)["input_ids"]
    prompt_ids = prompt_ids[-(args.max_len - len(answer_ids)):]          # keep the END of the log
    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids                       # learn only the answer
    return torch.tensor([ids]), torch.tensor([labels])

data = [json.loads(l) for l in open(args.train) if l.strip()]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
model.train(); step = 0
for ep in range(args.epochs):
    random.shuffle(data); total = 0
    for ex in data:
        ids, labels = encode(ex)
        loss = model(input_ids=ids.to(dev), labels=labels.to(dev)).loss
        loss.backward(); opt.step(); opt.zero_grad(); total += loss.item(); step += 1
    print(f"epoch {ep + 1}/{args.epochs}  mean loss {total / len(data):.4f}")
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print("saved adapter to", args.out)
