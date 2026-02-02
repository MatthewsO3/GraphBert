# Training & Evaluation Results

This document summarizes the step-by-step training and evaluation results for the Base Model across **Erlang** and **C++**, preserving all original metrics and values exactly as reported.

---

## STEP 1 — Base Evaluation `(0,0)`

### Base Model — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **1000**
- Total masked tokens: **43,221**

**Metrics**
- Top-1 Accuracy: **68.92%** (29,789 / 43,221)
- Top-5 Accuracy: **78.83%** (34,072 / 43,221)
- Perplexity: **79.5706**

---

### Base Model — Erlang

**Dataset**
- Total masked tokens: **17,251**

**Metrics**
- MLM Accuracy: **0.7313**
- MLM Loss: **2.1685**
- Perplexity: **8.7454**
- Top-5 Accuracy: **0.8245**
- Top-10 Accuracy: **0.8471**

---

## STEP 2 — 1 Epoch on Erlang `(1,0)`

**Training**
- Average losses:
  - Total: **0.9028**
  - MLM: **0.9624**
  - Edge: **0.9727**
  - Align: **0.7735**

### Evaluation — Erlang

- MLM Accuracy: **0.7683**
- MLM Loss: **1.7805**
- Perplexity: **5.9331**
- Top-5 Accuracy: **0.8623**
- Top-10 Accuracy: **0.8831**
- Total examples: **1000**
- Total masked tokens: **16,641**

### Evaluation — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **2500**
- Total masked tokens: **109,380**

**Metrics**
- Top-1 Accuracy: **63.42%** (69,372 / 109,380)
- Top-5 Accuracy: **75.05%** (82,091 / 109,380)
- Perplexity: **169.0928**

---

## STEP 3 — 1 Epoch on C++ `(1,1)`

**Training**
- Total: **1.933083**
- MLM: **1.478048**
- Edge: **0.455035**

**Validation**
- Total: **1.137038**
- MLM: **0.758761**
- Edge: **0.378276**

### Evaluation — Erlang

- MLM Accuracy: **0.7995**
- MLM Loss: **1.0417**
- Perplexity: **2.8340**
- Top-5 Accuracy: **0.8992**
- Top-10 Accuracy: **0.9237**
- Total examples: **1000**
- Total masked tokens: **16,641**

### Evaluation — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **2500**
- Total masked tokens: **109,380**

**Metrics**
- Top-1 Accuracy: **83.38%** (91,206 / 109,380)
- Top-5 Accuracy: **92.56%** (101,238 / 109,380)
- Perplexity: **4.8876**

---

## STEP 4 — 1 Epoch on Erlang `(2,1)`

**Training**
- Loss: **0.5981**
- Avg loss: **0.8975**
- Learning rate: **0.00e+00**
- MLM: **0.962**
- Edge: **0.953**
- Align: **0.778**

### Evaluation — Erlang

- MLM Accuracy: **0.7667**
- MLM Loss: **1.7938**
- Perplexity: **6.0121**
- Top-5 Accuracy: **0.8610**
- Top-10 Accuracy: **0.8850**
- Total examples: **1000**
- Total masked tokens: **16,493**

### Evaluation — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **2500**
- Snippets skipped: **2**
- Total masked tokens: **109,380**

**Metrics**
- Top-1 Accuracy: **63.40%** (69,350 / 109,380)
- Top-5 Accuracy: **75.11%** (82,151 / 109,380)
- Perplexity: **168.2548**

---

## STEP 5 — 1 Epoch on C++ `(2,2)`

**Training**
- Total: **1.906679**
- MLM: **1.452741**
- Edge: **0.453939**

**Validation**
- Total: **1.112505**
- MLM: **0.735843**
- Edge: **0.376662**

### Evaluation — Erlang

- MLM Accuracy: **0.7930**
- MLM Loss: **1.0783**
- Perplexity: **2.9396**
- Top-5 Accuracy: **0.8980**
- Top-10 Accuracy: **0.9233**
- Total examples: **1000**
- Total masked tokens: **16,699**

### Evaluation — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **2500**
- Total masked tokens: **109,380**

**Metrics**
- Top-1 Accuracy: **83.69%** (91,541 / 109,380)
- Top-5 Accuracy: **92.78%** (101,482 / 109,380)
- Perplexity: **4.6476**

---

## STEP 6 — 1 Epoch on Erlang `(3,2)`

**Training**
- Progress: **100% (679 / 679)**
- Time: **36:12**
- Iteration speed: **3.20s / it**
- Loss: **0.6283**
- Avg loss: **0.8936**
- Learning rate: **0.00e+00**
- MLM: **0.959**
- Edge: **0.941**
- Align: **0.781**

### Evaluation — Erlang

- MLM Accuracy: **0.7624**
- MLM Loss: **1.8289**
- Perplexity: **6.2272**
- Top-5 Accuracy: **0.8568**
- Top-10 Accuracy: **0.8778**
- Total examples: **1000**
- Total masked tokens: **16,451**

### Evaluation — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **2500**
- Total masked tokens: **109,380**

**Metrics**
- Top-1 Accuracy: **63.94%** (69,940 / 109,380)
- Top-5 Accuracy: **75.44%** (82,519 / 109,380)
- Perplexity: **158.3871**

---

## STEP 7 — 1 Epoch on C++ `(3,3)`

**Training**
- Total: **1.910117**
- MLM: **1.453342**
- Edge: **0.456775**

**Validation**
- Total: **1.112858**
- MLM: **0.736475**
- Edge: **0.376383**

### Evaluation — Erlang

- MLM Accuracy: **0.8036**
- MLM Loss: **1.0142**
- Perplexity: **2.7572**
- Top-5 Accuracy: **0.9032**
- Top-10 Accuracy: **0.9264**
- Total examples: **1000**
- Total masked tokens: **16,763**

### Evaluation — C++

**Dataset**
- Data file: `data/val.jsonl`
- Snippets evaluated: **2500**
- Snippets skipped: **2**
- Total masked tokens: **109,380**

**Metrics**
- Top-1 Accuracy: **83.63%** (91,473 / 109,380)
- Top-5 Accuracy: **92.80%** (101,501 / 109,380)
- Perplexity: **4.6865**


## Erlang Metrics Progression

------------------------------------------------------------------------------------
    Step Description   MLM        MLM Loss Perplexity   Top-5    Top-10    Masked
                       Accuracy                         Acc      Acc       Tokens
------ ------------- ---------- -------- ------------ -------- --------- -----------
       0 Base (STEP 1) 0.7313     2.1685   8.7454       0.8245   0.8471    17,251

       1 STEP 2 --- 1  0.7683     1.7805   5.9331       0.8623   0.8831    16,641
         Epoch Erlang                                                      
         (1,0)                                                             

       2 STEP 3 ---    0.7995     1.0417   2.8340       0.8992   0.9237    16,641
         After C++                                                         
         (1,1)                                                             

       3 STEP 4 --- 1  0.7667     1.7938   6.0121       0.8610   0.8850    16,493
         Epoch Erlang                                                      
         (2,1)                                                             

       4 STEP 5 ---    0.7930     1.0783   2.9396       0.8980   0.9233    16,699
         After C++                                                         
         (2,2)                                                             

       5 STEP 6 --- 1  0.7624     1.8289   6.2272       0.8568   0.8778    16,451
         Epoch Erlang                                                      
         (3,2)                                                             

       6 STEP 7 ---    0.8036     1.0142   2.7572       0.9032   0.9264    16,763
         After C++                                                         
         (3,3)                                                             
------------------------------------------------------------------------------------

------------------------------------------------------------------------

## C++ Metrics Progression

  -----------------------------------------------------------------------------
    Step Description   Top-1 Accuracy Top-5 Accuracy Perplexity   Masked Tokens
  ------ ------------- -------------- -------------- ------------ -------------
       0 Base (STEP 1) 68.92% (29,789 78.83% (34,072 79.5706      43,221
                       / 43,221)      / 43,221)                   

       1 STEP 2 ---    63.42% (69,372 75.05% (82,091 169.0928     109,380
         After Erlang  / 109,380)     / 109,380)                  
         (1,0)                                                    

       2 STEP 3 --- 1  83.38% (91,206 92.56%         4.8876       109,380
         Epoch C++     / 109,380)     (101,238 /                  
         (1,1)                        109,380)                    

       3 STEP 4 ---    63.40% (69,350 75.11% (82,151 168.2548     109,380
         After Erlang  / 109,380)     / 109,380)                  
         (2,1)                                                    

       4 STEP 5 --- 1  83.69% (91,541 92.78%         4.6476       109,380
         Epoch C++     / 109,380)     (101,482 /                  
         (2,2)                        109,380)                    

       5 STEP 6 ---    63.94% (69,940 75.44% (82,519 158.3871     109,380
         After Erlang  / 109,380)     / 109,380)                  
         (3,2)                                                    

       6 STEP 7 --- 1  83.63% (91,473 92.80%         4.6865       109,380
         Epoch C++     / 109,380)     (101,501 /                  
         (3,3)                        109,380)                    
-----------------------------------------------------------------------------

---
