# 🧬 Gemma-4-12B Bioinformatics QA — QLoRA Fine-Tuning

This repository contains a **research-grade Jupyter notebook** for fine-tuning **Gemma 4 12B (instruction-tuned)** on a bioinformatics Q&A dataset using **4-bit QLoRA + SFT (TRL)**. The resulting models are available on Hugging Face for both standard `transformers` usage and GGUF-based local inference.

- 🔗 **Fine-tuned model (Transformers):** https://huggingface.co/yashm/gemma4-12b-bioinfo  
- 🔗 **Quantized GGUF weights:** https://huggingface.co/yashm/gemma4-12b-bioinfo-GGUF

> ⚠️ This project is for **research and education** in bioinformatics and computational biology. It is **not** a medical device and must not be used for clinical decision-making.[web:2]

---

## ✨ What this repo gives you

- A **clean, end-to-end Jupyter notebook** that:
  - Verifies your environment (Python, CUDA, packages, Gemma 4 support)
  - Loads `google/gemma-4-12B-it` as the base model
  - Loads a bioinformatics Q&A dataset from Hugging Face
  - Formats data into **Gemma 4 chat-style messages**
  - Runs **QLoRA fine-tuning** with TRL’s `SFTTrainer`
  - Evaluates and saves **LoRA adapter weights** instead of the full 12B model
  - Includes a **qualitative inference section** with real bioinformatics questions

- **Reproducible training recipe** for the published models on Hugging Face, including hyperparameters tuned for a single 24 GB GPU.

---

## 🧩 Upstream models & dataset

- **Base model:** `google/gemma-4-12B-it` (multimodal, instruction-tuned)  
  Gemma 4 12B is released under the permissive **Apache-2.0 license**, allowing commercial use and modification as long as you comply with the license terms.[web:18][web:20][web:26]

- **Fine-tuned model (this work):**
  - `yashm/gemma4-12b-bioinfo` — standard `transformers` weights
  - `yashm/gemma4-12b-bioinfo-GGUF` — quantized GGUF files for `llama.cpp`, LM Studio, Ollama-compatible runtimes, and `llama-cpp-python`[web:2]

- **Training data (example):**
  - `yashm/bioinformatics-qa-dataset` — bioinformatics and computational biology Q&A pairs curated on Hugging Face.[web:19][web:23]

Make sure you respect the licenses/terms of **Gemma 4**, the **dataset(s)** you use, and any additional resources you integrate.[web:18][web:20][web:22]

---

## 🗂️ Repository structure

Typical layout for this repo:

- `gemed_git.ipynb` – main notebook:
  - Environment checks
  - Dataset loading & formatting
  - QLoRA fine-tuning with TRL
  - Evaluation & adapter export
  - Example inference prompts
- `requirements.txt` or `environment.yml` – environment spec (recommended)
- `LICENSE` – your chosen project license
- `README.md` – this file

You can rename the notebook if you prefer (e.g., `gemma4-bioinfo-qlora.ipynb`), but keep the execution order top-to-bottom.

---

## ⚙️ Environment & hardware

The notebook is written for:

- **Python:** 3.11
- **GPU:** single NVIDIA GPU (tested on RTX 4500 Ada 24 GB)
- **Key libraries (minimum versions):**
  - `torch` ≥ 2.0.0
  - `transformers` ≥ 5.0.0 (Gemma 4 support + `gemma4_unified` config)
  - `trl` ≥ 0.12.0
  - `peft` ≥ 0.14.0
  - `accelerate` ≥ 1.2.0
  - `bitsandbytes` ≥ 0.45.0
  - `datasets` ≥ 3.0.0

> ✅ The first notebook cells perform detailed **environment verification** (Python version, CUDA availability, GPU name & VRAM, package versions, and `gemma4_unified` support) and will fail early if something is misconfigured.

---

## 🚀 Getting started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/gemma4-12b-bioinfo-qlora.git
cd gemma4-12b-bioinfo-qlora
```

### 2. Create and activate a conda environment (recommended)

```bash
conda create -n gemma4 python=3.11 -y
conda activate gemma4
```

### 3. Install dependencies

If you provide a `requirements.txt`:

```bash
pip install -r requirements.txt
```

Otherwise, install the main pieces manually:

```bash
pip install "torch>=2.0.0" "transformers>=5.0.0" \
            "trl>=0.12.0" "peft>=0.14.0" "accelerate>=1.2.0" \
            "bitsandbytes>=0.45.0" "datasets>=3.0.0" "huggingface_hub"
```

Make sure `bitsandbytes` sees your GPU; otherwise, you may need a CUDA-compatible build.

### 4. Launch Jupyter

```bash
jupyter lab  # or: jupyter notebook
```

Open `gemed_git.ipynb`, set the kernel to your `gemma4` environment, and follow the numbered cells in order.

---

## 🧪 Training pipeline (high level)

The notebook walks through the following steps:

1. **Pre-flight checks**
   - Locks `CUDA_VISIBLE_DEVICES=0`
   - Checks Python version, packages, CUDA, GPU memory, and Gemma 4 config support

2. **Authentication & configuration**
   - Uses a **Hugging Face token (`HF_TOKEN`)** to:
     - Authenticate to the Hub
     - Confirm access to `google/gemma-4-12B-it` and your dataset
   - Defines constants:
     - `MODEL_ID = "google/gemma-4-12B-it"`
     - `DATASET_ID = "yashm/bioinformatics-qa-dataset"` (changeable)
     - `OUTPUT_DIR = "./gemma4-12b-bioinfo-qlora"`

3. **Dataset loading & inspection**
   - Loads the dataset from Hugging Face with `load_dataset`
   - Prints:
     - Available splits
     - Column names
     - Dataset sizes
     - A sample row for quick sanity-checking

4. **Formatting to Gemma 4 chat format**
   - Adapts Q&A columns (default: `question`, `answer`) into:
     ```python
     {
       "messages": [
         {"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": question},
         {"role": "model",  "content": answer},
       ]
     }
     ```
   - Ensures:
     - Role name is `model` (Gemma 4 uses `"model"` instead of `"assistant"`)
     - A small eval split is created (e.g., 95/5 train/test) if none exists

5. **Model loading with QLoRA**
   - Loads `google/gemma-4-12B-it` in **4‑bit NF4** using `BitsAndBytesConfig`
   - Wraps the model with **LoRA adapters** using PEFT
   - Config is tuned for a 24 GB GPU (batch size, gradient accumulation, sequence length)

6. **Supervised fine-tuning (SFT via TRL)**
   - Uses `SFTTrainer` with:
     - Chat template from `tokenizer.apply_chat_template`
     - Training/eval splits from the formatted dataset
   - Logs training loss and evaluation metrics at regular intervals

7. **Evaluation & adapter export**
   - Runs `trainer.evaluate()` on the held-out split
   - Saves only the **LoRA adapter** and tokenizer to:
     - `./gemma4-12b-bioinfo-qlora/final-adapter`

8. **Qualitative inference**
   - Defines a `generate_answer(question: str)` helper:
     - Builds a Gemma 4 chat prompt
     - Runs generation on the fine-tuned model
     - Decodes and prints the answer
   - Includes sample bioinformatics questions (e.g., BLAST vs HMMER, Phred scores, GATK workflows, RNA‑seq pipelines)

---

## 🔎 Using the published models

### Option A — `transformers` (standard fine-tuned model)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "yashm/gemma4-12b-bioinfo"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an expert bioinformatics assistant."},
    {"role": "user", "content": "Explain CpG islands and their biological relevance."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9,
)

answer = tokenizer.decode(outputs[inputs["input_ids"].shape:], skip_special_tokens=True)[1]
print(answer.strip())
```

> 💡 For multimodal use (image or other inputs), refer to the latest Gemma 4 12B developer docs and `AutoModelForImageTextToText` guidance.[web:5][web:14]

### Option B — GGUF (`llama.cpp`, LM Studio, Ollama, llama-cpp-python)

The GGUF repo (`yashm/gemma4-12b-bioinfo-GGUF`) contains multiple quantizations such as a 4‑bit variant for efficient local inference.[web:2]

**Example with `llama.cpp` (CLI):**

```bash
./llama-cli \
  -m ./gemma4-12b-bioinfo-Q4_K_M.gguf \
  -p "<|turn>user\nExplain the role of CRISPR-Cas9 in genome editing.\n<|turn>model\n" \
  -n 512 \
  -c 2048 \
  --temp 0.2 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

**Example with `llama-cpp-python`:**

```python
from llama_cpp import Llama

llm = Llama(
    model_path="gemma4-12b-bioinfo-Q4_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # set 0 for CPU-only
    verbose=False,
)

question = "Explain the significance of CRISPR-Cas9 in functional genomics."
prompt = f"<|turn>user\n{question}<|turn>model\n"

output = llm(
    prompt,
    max_tokens=512,
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
    stop=["<|turn>user", "<eos>"],
)

print(output["choices"]["text"].strip())
```

---

## 💡 Tips for adapting this notebook

- **Change the dataset**
  - Update `DATASET_ID` and the `QUESTION_COL` / `ANSWER_COL` names in the dataset formatting cell.
  - Re-run from the dataset loading cell downwards.

- **Adjust system prompt**
  - The default `SYSTEM_PROMPT` targets **bioinformatics** (genomics, proteomics, NGS pipelines, tools like BLAST, HMMER, BWA, GATK, DESeq2, etc.).
  - Customize it for your domain (e.g., structural biology, cheminformatics) for better steering.

- **Tune for your GPU**
  - Lower `MAX_SEQ_LEN` or increase gradient accumulation (`GRAD_ACCUM`) if you run out of VRAM.
  - Consider smaller batch sizes or more aggressive quantization if you are on 16 GB GPUs.

- **Experiment with sampling**
  - For more deterministic answers, use **lower temperature**.
  - For more diverse exploration, slightly increase temperature/top‑p and compare results qualitatively.

- **Logging & checkpoints**
  - Enable more frequent evaluation or checkpoint saving in `SFTConfig` if you plan longer training runs.
  - Use `trainer.save_model()` or push adapters directly to the Hub when you are satisfied with performance.

---

## ⚖️ Safety, limitations & responsible use

- The model is optimized for **bioinformatics and computational biology assistance**, not for general medicine or clinical care.[web:2][web:19]
- Outputs may be **incorrect, incomplete, or outdated**; always cross-check against:
  - Primary literature  
  - Curated databases (NCBI, Ensembl, UniProt, PDB, etc.)  
  - Domain experts
- Do **not** use this model for:
  - Diagnosing patients
  - Choosing treatments or drugs
  - Any task that requires regulatory approval or clinical validation

By using this repository and the associated models, you agree to take full responsibility for verifying outputs and for complying with all applicable laws, regulations, and licensing terms.[web:2][web:18]

---

## 📜 License & attribution

- **Base model:** Gemma 4 12B weights are released by Google under the **Apache-2.0 license**, which permits commercial and derivative use with appropriate attribution.[web:18][web:20][web:26]
- **This repository:** Choose and include a license file (e.g., Apache-2.0, MIT) and update this section accordingly.
- **Datasets:** Respect the licenses and usage terms of `yashm/bioinformatics-qa-dataset` and any other datasets you plug into the notebook.[web:19][web:23]

When you use or publish work based on this repo, please consider citing:

- The Gemma 4 model announcement and technical documentation  
- The dataset(s) you trained on  
- This GitHub repository and the associated Hugging Face model cards

---

## 🙌 Acknowledgements

- **Gemma 4** team and contributors for releasing powerful, Apache‑2.0‑licensed models to the community.[web:18][web:20]
- **Hugging Face** for hosting models, datasets, and making fine-tuning workflows accessible.
- The open-source bioinformatics and ML communities whose tools and datasets made this work possible.
