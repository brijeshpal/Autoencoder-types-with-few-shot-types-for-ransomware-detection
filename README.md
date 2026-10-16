# Few-Shot Ransomware Detection with Variational Autoencoder + Meta-Learning

This notebook (`few_shot_with_autoencoders_types_Project.ipynb`) studies **few-shot binary malware detection** (benign vs. malware) on tabular PE/behavioral features. The idea is to detect malware when only a handful of labelled samples (K per class) are available.

The approach is a two-stage pipeline:

1. **Unsupervised feature extraction:** an autoencoder compresses the 72 input features into a 64-dimensional latent space.
2. **Few-shot classification:** a meta-learner (MAML and variants) is trained on many small episodes so it can adapt to a new task from only K labelled examples per class.

The notebook compares several autoencoder types and several few-shot learners, sweeps K from 1 to 50, and ends with a cleaned-up final pipeline that uses a proper train/test split.

---

## Pipeline

```
Dataset.csv (72 features)
      │
      ▼
Preprocessing ── drop IDs (md5, sha1, Category, Family)
      │          hex strings → int, categoricals → LabelEncoder
      │          Min-Max scaling to [0, 1]
      ▼
Autoencoder (72 → 256 → 128 → 64 → 128 → 256 → 72), MSE loss
      │
      ▼  64-d latent vector z (VAE: mean μ)
      │
Episode sampler: 2-way K-shot support set + query set
      │
      ▼
Meta-learner (MAML / MAML++ / CAVIA / ProtoNet / RelationNet)
      │
      ▼
Accuracy, F1, Precision, Recall, MCC, confusion matrix
```

---

## Notebook structure

Each code cell is **self-contained**: it re-imports libraries and redefines `Config`, the models and the helpers, so cells can be run independently.

| Cell | Purpose | Status |
|------|---------|--------|
| 1 | Prototype on the **EMBER 2018** dataset (2,381 features, larger AE: 2381 → 1024 → 512 → 128 → 64) with AE + MAML for K ∈ {1, 5, 20} | Not run: fails with `ModuleNotFoundError: No module named 'lief'` |
| 2 | Compare **Standard, Denoising, Sparse and Variational AEs** as feature extractors, each followed by MAML (K = 5) | Ran |
| 3 | Same AE comparison, refactored (uses `torch.func.functional_call`, gradient clipping, VAE uses μ) | Ran |
| 4 | Encoder × few-shot learner: **Sparse AE / VAE × MAML / ProtoNet / RelationNet** (K = 5) | Ran |
| 5 | Encoder × MAML variants: **Sparse AE / VAE × MAML / MAML++ / CAVIA** (K = 5) | Ran |
| 6 | Full **K-shot sweep** K ∈ {1, 2, 5, 10, 20, 50} for Sparse AE / VAE × MAML / MAML++ / CAVIA | Ran |
| 7 | **Final AFE-MAML pipeline**: seeded, stratified 75/25 train/test split, scaler fit on train only, 100 AE epochs, 1,000 meta-episodes, 1,000 query samples per class, confusion matrices and a summary table | Not run: fails with `NameError: name '__file__' is not defined` (see fix below) |

---

## Requirements

- Python 3.9+
- PyTorch 2.0+ (needed for `torch.func.functional_call`)
- numpy, pandas, scikit-learn
- A GPU is recommended (the notebook was run on a Colab T4). CPU works but is slower.

```bash
pip install torch numpy pandas scikit-learn
```

Cell 1 (EMBER) additionally needs:

```bash
pip install git+https://github.com/elastic/ember.git lightgbm lief
```

EMBER's feature extractor was written against older LIEF releases, so you may need to pin a compatible `lief` version. The EMBER download and vectorisation step is large and RAM-heavy. Cell 1 is optional; the main experiments do not use it.

---

## Dataset

Place the dataset at `/content/Dataset.csv` (the default path in cells 2–6).

Expected format:

- **Identifier/metadata columns**, dropped before training: `md5`, `sha1`, `Category`, `Family`
- **Label column:** `Class`. In cells 2–6, `Benign` (case-insensitive) maps to 0 and every other value maps to 1. Cell 7 uses `LabelEncoder`, which assumes exactly two class values.
- **72 feature columns.** Some may be hex strings (e.g. `0x108ec`); these are converted to integers. Non-numeric columns that cannot be converted are label-encoded.

Cells 2–6 pad or truncate to 72 features. Cell 7 instead **asserts** that exactly 72 features remain after preprocessing.

---

## How to run (Google Colab)

1. Open the notebook in Colab and select **Runtime → Change runtime type → GPU**.
2. Upload `Dataset.csv` to `/content/`.
3. Skip cell 1 unless you want the EMBER experiment.
4. Run any of cells 2–6 for the comparison experiments.
5. For cell 7, first fix the dataset path. `__file__` does not exist inside a notebook, so replace:

   ```python
   DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset.csv')
   ```

   with:

   ```python
   DATASET_PATH = '/content/Dataset.csv'
   ```

   (If you export cell 7 as a standalone `.py` script, the original line works as written.)

---

## Models

### Autoencoders (feature extractors)

All variants share the encoder 72 → 256 → 128 → 64 and decoder 64 → 128 → 256 → 72, with ReLU activations.

| Variant | Training objective |
|---------|-------------------|
| Standard AE | MSE reconstruction |
| Denoising AE | Gaussian noise (σ = 0.1) added to the input; reconstruct the clean input |
| Sparse AE | MSE + 1e-4 × L1 penalty on the latent code |
| Variational AE | MSE (sum) + 0.001 × KL divergence, averaged per batch. The mean **μ** (not a sampled z) is used as the feature vector for classification. |

### Few-shot learners

| Method | Description |
|--------|-------------|
| **MAML** | MLP 64 → 128 → 64 → 2. Inner loop: 5 SGD steps (α = 0.01) on the support set. Outer loop: Adam (β) on the query loss, with second-order gradients. |
| **MAML++** (simplified) | MAML plus learnable per-parameter-tensor inner learning rates (LSLR) and a multi-step loss (query loss at every inner step, weighted linearly towards later steps). |
| **CAVIA** | Inner loop adapts only a 50-d context vector (lr = 0.1) concatenated to the input; network weights are updated only in the outer loop. |
| **ProtoNet** | Class prototypes = mean latent vector per class; nearest prototype by Euclidean distance. No training: it uses the AE embeddings directly. |
| **RelationNet** | Concatenates [prototype, query] (128-d) → 128 → 64 → 32 → 1 (sigmoid) relation score, trained with MSE against one-hot targets. |

---

## Hyperparameters

| Parameter | Cells 2–6 | Cell 7 (final) |
|-----------|-----------|----------------|
| Latent dim | 64 | 64 |
| AE learning rate / epochs / batch | 0.001 / 50 / 64 | 0.001 / 100 / 64 |
| Meta (outer) learning rate β | 0.005 | 0.001 |
| Inner learning rate α | 0.01 | 0.01 |
| Inner steps | 5 | 5 |
| Meta-training episodes | 500 | 1,000 |
| Meta-testing episodes | 50–100 | 50 |
| Query samples per class | 100 | 1,000 |
| Train/test split | none | stratified 75/25 |
| Random seed | not set | 42 |

---

## Known issues and limitations

- **Cell 1:** the `ember` package needs `lief`, which is not installed by the cell.
- **Cell 7:** `__file__` is undefined in notebooks; see the fix in *How to run*.
- **Data leakage in cells 2–6:** the scaler and autoencoder are fit on the full dataset, and meta-training and meta-testing episodes are drawn from the same pool of samples. These results are likely optimistic. Cell 7 addresses this with a train/test split and a train-only scaler, but its results have not been produced yet.
- **Sampling with replacement in cell 2:** support and query indices are drawn independently, so the same sample can appear in both sets of an episode.
- **Same task at train and test time:** every episode is the same benign-vs-malware task, so this setup measures low-data adaptation rather than generalisation to unseen task types (e.g. unseen malware families). Splitting episodes by `Family` would make a stronger few-shot benchmark.
- **No seeds in cells 2–6** and single runs only, so small differences between methods (≈ 1%) may be within run-to-run variance. Cell 7 reports standard deviations for accuracy and F1.
- **MAML++** here is a simplified version (LSLR + multi-step loss only); it omits other parts of the original method such as per-step batch-norm statistics and derivative-order annealing.

---

## Possible next steps

- Run the fixed cell 7 to get leakage-free results and confusion matrices.
- Use cell 7's split-and-scale approach for the encoder/learner comparisons in cells 2–6.
- Repeat each experiment over several seeds and report mean ± std.
- Build episodes from held-out malware families to test generalisation to new threats.
- Fine-tune the encoder jointly with the meta-learner instead of freezing it.
