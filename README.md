# Product Description Generation with GPT-2

Conditional text generation: given the structured facts of an e-commerce product (title, features, details, optional category), a fine-tuned GPT-2 writes a marketing description.

## Overview

| Step | What is done |
|---|---|
| Dataset | About 100,000 Amazon products in 10 categories, each with a title, features, details and a real description |
| Preprocessing | A parser splits the raw text into title, features and detail pairs; descriptions are cleaned (HTML, links, length, duplicates) |
| Representation | Five field marker tokens (category, title, features, details, description start) with a token budget per field |
| Modeling | GPT-2 fine-tuned with AdamW, warm-up and linear decay, gradient accumulation, early stopping and mixed precision |
| Evaluation | BLEU, chrF, perplexity, faithfulness checks and a zero-shot GPT-2 baseline |
| Application | Streamlit demo that builds the prompt exactly as in training |

### Key design choices

- **Field-aware encoding.** Each field has its own marker token and token budget, so a long feature list cannot push the details out of the input.
- **Loss on the description only.** Prompt and padding positions are masked, so the model is graded only on writing the description.
- **Optional category.** The category is hidden in 10% of the training prompts, so the model also works without it.
- **No leakage.** Every validation and test product is removed from the training data.
- **Decoding chosen on validation data.** Greedy, beam search and top-p sampling are compared on validation products only. A repeated 3-word phrase is blocked in the generated text, while copying words from the product facts remains allowed.

### Faithfulness checks

- `invented_number_rate`: share of outputs containing a number that is not in the product facts
- `detail_recall`: share of detail values (color, size, material...) that appear in the output
- `title_word_recall`: share of title words that appear in the output

## Repository structure

```
.
├── app.py                  Streamlit demo
├── requirements.txt
├── notebooks/
│   └── gpt2_product_descriptions_v2.ipynb   data preparation, training, evaluation
└── docs/
    └── Generating_Product_Descriptions_with_GPT-2.pptx   presentation
```

The model weights and the dataset are not stored in this repository.

## Training and evaluation

The notebook is written for Kaggle (GPU T4, internet on for the first model download). It trains the model, saves the best checkpoint together with `product_prompt.py` and `run_config.json` in a folder named `gpt2-tokens-best/`, and writes the evaluation results to `results_v2.json`.

## Running the demo

1. Place the `gpt2-tokens-best/` folder produced by the notebook next to `app.py`.
2. Install the dependencies and start the app:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app lets the user enter a title, features, details and an optional category. It shows a warning when the generated text contains a number that is not in the input.

## Limitations

- The model can state facts that are not in the input, so generated descriptions must be checked before they are published.
- BLEU is a weak signal for marketing text, because two valid descriptions share few n-grams. It is reported together with perplexity and the faithfulness checks.
- Some outputs repeat the input features closely instead of rewriting them.

## Data

Amazon product data. The dataset is not redistributed here.
