"""Generate a harder, deconfounded sample set into ./samples_claude/.

Original samples/*.txt had a confound: every math prompt starts with a
number-word ("Ten plus five...") and every "others" prompt starts with an
animal-word ("Cats love fish..."). A layer-0 MLP is mostly a token-identity
detector, so the original 100% LOOCV accuracy could trivially come from
"is the first token a number-word", not from anything math-specific.

This script removes that confound: BOTH classes now start with the same
pool of number-words. Math prompts contain an arithmetic operator
(plus/minus/times) + "equals to"; others prompts use the same numbers as
plain quantifiers over animal subjects, with no arithmetic structure.

Extraction matches hui_gpt2_1.ipynb cells 20-23 exactly:
block 0 -> ln_2 -> mlp.c_fc -> mlp.act (GELU)  => (tokens, 2816) matrix,
saved with np.savetxt, one file per prompt, plus a samples_claude.txt
metadata index in the same "|prompt| - name; token_count" format as the
original samples/samples.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "./weiser/101M-0.4"
OUT_DIR = Path(__file__).resolve().parent.parent / "samples_claude"
OUT_DIR.mkdir(exist_ok=True)

# Same number-words appear in BOTH classes -> kills the "first token = number"
# shortcut that the original dataset had.
NUMBERS = ["Five", "Ten", "Seven", "Six", "Eight", "Nine", "Four", "Twelve", "Two", "Three"]

MATH_PROMPTS = [
    "Five plus two equals to ",
    "Ten minus three equals to ",
    "Seven plus one equals to ",
    "Six times two equals to ",
    "Eight minus four equals to ",
    "Nine plus three equals to ",
    "Four times three equals to ",
    "Twelve minus six equals to ",
    "Two times five equals to ",
    "Three plus three equals to ",
]

OTHER_PROMPTS = [
    "Five cats slept all day because of ",
    "Ten dogs ran in the park because of ",
    "Seven birds flew away because of ",
    "Six frogs jumped into water because of ",
    "Eight bees flew around because of ",
    "Nine cows grazed quietly because of ",
    "Four horses galloped fast because of ",
    "Twelve monkeys climbed trees because of ",
    "Two spiders spun webs because of ",
    "Three bears searched for honey because of ",
]


def extract_layer0_mlp_postact(model, tokenizer, prompt: str) -> np.ndarray:
    tokens = tokenizer.tokenize(prompt)
    ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    with torch.no_grad():
        tok_emb = model.transformer.wte(ids)
        pos_emb = model.transformer.wpe(torch.arange(ids.shape[1]).unsqueeze(0))
        x = tok_emb + pos_emb

        block = model.transformer.h[0]
        x_norm = block.ln_1(x)

        attn_out = block.attn(x_norm)[0]
        hidden_after_attn = x + attn_out

        x_norm2 = block.ln_2(hidden_after_attn)
        h = block.mlp.c_fc(x_norm2)
        h = block.mlp.act(h)  # post-GELU, (1, tokens, 2816) -- matches samples/*.txt

    return h[0].detach().cpu().to(torch.float32).numpy(), len(tokens)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    meta_lines = []
    for label, prompts in [("math", MATH_PROMPTS), ("others", OTHER_PROMPTS)]:
        for i, prompt in enumerate(prompts):
            matrix, n_tok = extract_layer0_mlp_postact(model, tokenizer, prompt)
            name = f"{label}_{i}"
            np.savetxt(OUT_DIR / f"{name}.txt", matrix)
            meta_lines.append(f"|{prompt.strip()} | - {name}; {n_tok}")
            print(f"{name:10s} tokens={n_tok:2d} shape={matrix.shape} prompt={prompt!r}")

    (OUT_DIR / "samples_claude.txt").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    print(f"\nSaved {len(meta_lines)} samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
