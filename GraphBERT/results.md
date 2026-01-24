## STEP 1: Fine-Tuning on Erlang

**Configuration**
- Batch size: `32`
- Training samples: `40` functions
- Validation samples: `5` functions
- Epochs: `3`

### Training & Validation Losses

| Epoch | Train Total | Train MLM | Train Edge | Train Align | Val Total | Val MLM | Val Edge | Val Align |
|------:|------------:|----------:|-----------:|------------:|----------:|--------:|---------:|----------:|
| 1 | 1.9575 | 2.1919 | 2.1693 | 1.5113 | 1.9322 | 2.0639 | 2.6138 | 1.1190 |
| 2 | **1.7960** | **1.6933** | 2.4182 | **1.2765** | **1.5909** | **1.1496** | 2.5967 | **1.0264** |
| 3 | 1.8460 | 1.9644 | **2.0218** | 1.5520 | 1.7077 | 1.5471 | **2.4848** | 1.0911 |

---

## STEP 2: Evaluation - Erlang Model

### Erlang MLM Metrics

| Metric | Value |
|------|------:|
| MLM Accuracy | **67.97%** |
| MLM Loss | 2.0032 |
| Perplexity | 7.4130 |
| Top-5 Accuracy | 81.05% |
| Top-10 Accuracy | 84.31% |
| Total Masked Tokens | 153 |

### Evaluation on C++ (Before C++ Training)

| Metric | Value |
|------|------:|
| Snippets Evaluated | 100 |
| Masked Tokens | 2500 |
| Top-1 Accuracy | **61.92%** |
| Top-5 Accuracy | 76.56% |
| Perplexity | **223.05** |

---

## STEP 3: Continued Fine-Tuning on C++

**Configuration**
- Training samples: `500`
- Batch size: `32`
- Epochs: `3`
- Learning rate decay to zero
- Early stopping patience: `3`

### Training Progress

| Epoch | Train Total | Train MLM | Train Edge | Val Total | Val MLM | Val Edge | LR |
|------:|------------:|----------:|-----------:|----------:|--------:|---------:|----:|
| 1 | 4.2130 | 3.5223 | 0.6907 | 3.8606 | 3.2590 | 0.6016 | 1.51e-5 |
| 2 | 3.6021 | 3.0351 | 0.5670 | 3.2687 | 2.7421 | 0.5266 | 7.57e-6 |
| 3 | **3.2414** | **2.7300** | **0.5114** | **2.9592** | **2.4560** | **0.5032** | 0.00 |

---

## STEP 4: Final Evaluation - C++ Model

### C++ MLM Metrics

| Metric | Value |
|------|------:|
| Snippets Evaluated | 100 |
| Masked Tokens | 2500 |
| Top-1 Accuracy | **62.32%** |
| Top-5 Accuracy | 76.60% |
| Perplexity | **233.09** |

### Erlang Retention After C++ Fine-Tuning

| Metric | Value |
|------|------:|
| MLM Accuracy | **77.55%** |
| MLM Loss | 1.3108 |
| Perplexity | **3.7090** |
| Top-5 Accuracy | 85.71% |
| Top-10 Accuracy | 87.76% |
| Masked Tokens | 147 |

---

##
<img width="3913" height="3361" alt="graphcodebert_all_metrics" src="https://github.com/user-attachments/assets/3bca09fa-a818-4e22-89fd-7969ebca6f57" />
<img width="3524" height="2670" alt="combined" src="https://github.com/user-attachments/assets/3e4176cc-1edf-4f45-8c79-12de7f8231f5" />

