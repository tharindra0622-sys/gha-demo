#!/usr/bin/env python3
"""
Step 3 - compare the SAME small model before and after post-training on runs it has never trained on.

    python3 evaluate_model.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter post_training/adapter
Output: post_training/results.json and a printed table (copy it into the thesis).
"""
import json, argparse
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from common import SYSTEM_PROMPT, parse_category

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--adapter", default="post_training/adapter")
ap.add_argument("--data-dir", default="post_training/data")
ap.add_argument("--max-len", type=int, default=2048)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if dev == "cuda" else torch.float32)
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev).eval()
tuned = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev), args.adapter).eval()


@torch.no_grad()
def predict(model, prompt):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)[-args.max_len:]
    out = model.generate(torch.tensor([ids]).to(dev), max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
    return parse_category(tok.decode(out[0][len(ids):], skip_special_tokens=True))


results = {}
for split in ["test_seen", "test_heldout"]:
    path = f"{args.data_dir}/{split}.jsonl"
    try:
        rows = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        continue
    if not rows:
        continue
    res = {}
    for name, model in [("base", base), ("post_trained", tuned)]:
        hit, per = 0, defaultdict(lambda: [0, 0])
        for r in rows:
            ok = predict(model, r["prompt"]) in r["accepted"]
            hit += ok; per[r["pipeline"]][0] += ok; per[r["pipeline"]][1] += 1
        res[name] = {"accuracy": hit / len(rows), "n": len(rows), "per_pipeline": {k: f"{a}/{b}" for k, (a, b) in per.items()}}
    results[split] = res
    print(f"\n{split} (n={len(rows)})")
    for name in res:
        print(f"  {name:13s} accuracy {res[name]['accuracy']:.1%}   {res[name]['per_pipeline']}")
json.dump(results, open("post_training/results.json", "w"), indent=2)
