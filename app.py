"""Streamlit demo for the v2 model (GPT-2 with field marker tokens).

Run:  streamlit run app.py
Folder layout (copy `gpt2-tokens-best/` from the Kaggle notebook output):

    gpt2-product-descriptions/
      app.py
      gpt2-tokens-best/      <- model, tokenizer, run_config.json, product_prompt.py
"""
import importlib.util
import json
import os
import re

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource(show_spinner="Loading the model...")
def load(model_dir):
    # Use the exact prompt module saved by the training notebook
    spec = importlib.util.spec_from_file_location("product_prompt", os.path.join(model_dir, "product_prompt.py"))
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    cfg = json.load(open(os.path.join(model_dir, "run_config.json")))
    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
    return pp, cfg, tok, model, device


def numbers(s):
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", str(s))}


st.set_page_config(page_title="Product description generator", page_icon="🛍️")
st.title("Product description generator")
st.caption("Fine-tuned GPT-2 trained on Amazon product data. Always check the output against the facts.")

with st.sidebar:
    st.header("Settings")
    model_dir = st.text_input("Model folder", "gpt2-tokens-best")
    ready = os.path.isfile(os.path.join(model_dir, "run_config.json")) and os.path.isfile(os.path.join(model_dir, "product_prompt.py"))
    if ready:
        pp, cfg, tok, model, device = load(model_dir)
        base = dict(cfg["gen_cfg"])
        beams = st.slider("Beams (1 = greedy, faster)", 1, 6, int(base.get("num_beams", 4)))
        min_new = st.slider("Minimum new tokens", 0, 80, int(base.get("min_new_tokens", 40)))
        len_pen = st.slider("Length penalty", 0.5, 2.0, float(base.get("length_penalty", 1.2)), 0.1)
        rep_pen = st.slider("Repetition penalty", 1.0, 1.5, 1.0, 0.05, help="1.0 = the setting used in the evaluation.")
    else:
        st.error("Folder not found, or run_config.json / product_prompt.py is missing.")

SAMPLE = {
    "title": "Wireless Bluetooth Headphones",
    "features": "Bluetooth 5.3\n30-hour battery life\nFoldable design",
    "details": "Color: Black",
    "category": "Electronics",
}
if "form" not in st.session_state:
    st.session_state.form = {"title": "", "features": "", "details": "", "category": "(none)"}
if st.button("Fill with an example product"):
    st.session_state.form = dict(SAMPLE)

f = st.session_state.form
title = st.text_input("Title", f["title"])
features = st.text_area("Features (one per line)", f["features"], height=140)
details = st.text_area("Details (one 'Name: value' per line)", f["details"], height=100)
cats = ["(none)"] + (cfg.get("categories", []) if ready else [])
category = st.selectbox("Category (optional)", cats, index=cats.index(f["category"]) if f["category"] in cats else 0)

if st.button("Generate description", type="primary"):
    if not ready:
        st.error("Set a valid model folder in the sidebar first.")
    elif not title.strip():
        st.warning("Enter at least a title.")
    else:
        rec = pp.make_record(title, features, details)
        gen = dict(base, num_beams=beams, min_new_tokens=min_new, length_penalty=len_pen, repetition_penalty=rep_pen)
        if beams == 1:                        # these options only apply to beam search
            gen.pop("early_stopping", None)
            gen.pop("length_penalty", None)
        cat = None if category == "(none)" else category
        with st.spinner(f"Generating on {device}..."):
            text = pp.describe(model, tok, rec, cat, cfg["encoding"], gen, cfg["max_prompt_tokens"], device=device)
        st.subheader("Generated description")
        st.write(text)
        st.caption(f"{len(text.split())} words")
        facts = pp.facts_text(rec)
        new = {n for n in numbers(text) - numbers(facts) if len(n.replace(".", "")) >= 2}
        if new:
            st.warning("Numbers in the text that are not in your input (check them): " + ", ".join(sorted(new)))
        with st.expander("Prompt sent to the model"):
            ids = pp.build_prompt_ids(tok, rec, cat, cfg["encoding"], cfg["max_prompt_tokens"])
            st.code(tok.decode(ids))
