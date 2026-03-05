---
marp: true
theme: default
paginate: true
size: 16:9
math: mathjax
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 36px;
    color: #1a5276;
  }
  h2 {
    font-size: 30px;
    color: #2c3e50;
  }
  table {
    font-size: 18px;
  }
  code {
    font-size: 16px;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
  .small {
    font-size: 18px;
  }
  .highlight {
    background: #eaf2f8;
    padding: 0.5rem 1rem;
    border-left: 4px solid #2980b9;
    border-radius: 4px;
  }
---

# ATIA: An Agentic RAG Architecture with Hybrid Knowledge-Gap Reasoning for Adaptive Technical Interview Assessment

**Research Proposal — Expert Systems with Applications (Q1, IF 10.48)**

Author: Phuong Bui

---

## Vấn đề nghiên cứu

Hệ thống phỏng vấn kỹ thuật tự động dựa trên LLM có **hai hạn chế cốt lõi**:

### 1. Tạo câu hỏi tĩnh

- Không thích ứng theo trạng thái kiến thức của ứng viên
- Bỏ sót lỗ hổng kiến thức quan trọng

### 2. Đánh giá dựa trên parametric knowledge

- LLM "ảo giác" khi đánh giá — nguồn sai lệch trong bối cảnh đánh giá quan trọng
- Không có grounding với tài liệu tham khảo đáng tin cậy

> **Mục tiêu:** Xây dựng kiến trúc multi-agent RAG thích ứng, có cơ sở lý thuyết, và có thể truy vết cho đánh giá phỏng vấn kỹ thuật.

---

## Câu hỏi nghiên cứu

| ID      | Câu hỏi                                                                                 | Đo bằng                           |
| ------- | --------------------------------------------------------------------------------------- | --------------------------------- |
| **RQ1** | Multi-agent có cải thiện chất lượng đánh giá so với monolithic LLM và single-agent RAG? | AAS, hallucination rate           |
| **RQ2** | Hybrid gap reasoning (graph + neural) có tốt hơn purely neural?                         | Gap Detection F1                  |
| **RQ3** | Policy hình thức hóa có tạo interview trajectory tốt hơn?                               | Topic coverage, depth progression |
| **RQ4** | Differentiated retrieval có cải thiện grounding?                                        | Grounding rate, factual accuracy  |
| **RQ5** | ATIA có duy trì chất lượng với ứng viên "khó"?                                          | Edge Case Handling Rate           |

---

## Kiến trúc tổng quan — ATIA

```
┌───────────────────────────────────────────────────────────────────┐
│                    Interview Context Object (ICO)                 │
│  ┌────────────────┐  ┌───────────────┐  ┌──────────────────────┐  │
│  │Candidate Model │  │Interview State│  │Knowledge Context     │  │
│  │- demonstrated  │  │- phase        │  │- retrieved_refs      │  │
│  │- gaps (PQ)     │  │- history      │  │- grounding_cache     │  │
│  │- misconceptions│  │- coverage_map │  │- prerequisite_graph  │  │
│  └────────────────┘  └───────────────┘  └──────────────────────┘  │
└──────────────┬──────────────┬─────────────┬───────────────────────┘
               │              │             │
    ┌──────────▼───┐   ┌──────▼──────┐  ┌───▼────┐   ┌──────────┐
    │ Response     │──>│ Knowledge   │─>│Question│──>│Grounded  │
    │ Analyst (RA) │   │ Gap Analyzer│  │Strategy│   │Question  │
    │              │   │ (KGA)       │  │(QSA)   │   │Gen (GQG) │
    └──────────────┘   └─────────────┘  └────────┘   └──────────┘
     Assertion-based    Graph-based      Pattern-      Content-based
     retrieval          retrieval        based         retrieval
```

---

## Đóng góp chính — 4 contributions

<div class="columns">
<div>

### C1: Multi-agent + Differentiated Retrieval

Mỗi agent có retrieval strategy riêng thay vì dùng chung một pipeline RAG duy nhất.

### C2: Hybrid Knowledge-Gap Reasoning

Kết hợp graph traversal (symbolic) + LLM interpretation (neural). Phát hiện "surface knowledge" — biết topic nhưng thiếu prerequisites.

</div>
<div>

### C3: Formalized Assessment Policy

QSA hoạt động trên state-action policy tường minh, mô hình hóa như POMDP approximation — interpretable, ablation-friendly.

### C4: Evaluation Framework

Benchmark phỏng vấn mô phỏng với ground truth xác định + expert validation + multi-backbone generalizability.

</div>
</div>

---

## Interview Context Object (ICO) — Shared State

<div class="columns">
<div>

### Candidate Model

- `demonstrated`: Map topic → level (NONE → DEEP) + evidence + confidence
- `gaps`: PriorityQueue theo severity
- `misconceptions`: claim + correction + source

### Interview State

- `phase`: WARMUP → PROBING → DEEP_DIVE → SYNTHESIS
- `questions_asked`: lịch sử đầy đủ
- `coverage`: tỷ lệ bao phủ mỗi topic
- `confirmed_gaps`: dừng probe topics đã xác nhận

</div>
<div>

### Knowledge Context

- `grounding_cache`: cache kết quả verification
- `prerequisite_graph`: đồ thị phụ thuộc

### Thiết kế

- Tất cả agents đọc/ghi vào ICO
- Đảm bảo specialization boundaries
- Full observability & traceability

</div>
</div>

---

## Agent 1: Response Analyst (RA)

**Mục đích:** Phân loại phản hồi ứng viên + đánh giá nội dung qua claim-level grounding.

### Bước 0 — Response Classification

| Loại              | Xử lý                          | Flow                 |
| ----------------- | ------------------------------ | -------------------- |
| **SUBSTANTIVE**   | Full claim extraction pipeline | RA → KGA → QSA → GQG |
| **PARTIAL**       | Extract reduced claim set      | RA → KGA → QSA → GQG |
| **DONT_KNOW**     | Mark gap, confidence = HIGH    | RA → KGA → QSA → GQG |
| **CLARIFICATION** | Không update candidate model   | RA → GQG (rephrase)  |
| **OFF_TOPIC**     | Không update candidate model   | RA → QSA → GQG       |

### Retrieval

- **Assertion-based**: claim → verification query → retrieve top-5 + cross-encoder reranking
- Corpus: official docs, textbook excerpts, verified Q&A

---

## Agent 2: Knowledge Gap Analyzer (KGA)

**Mục đích:** Chẩn đoán trạng thái kiến thức qua hybrid symbolic + neural reasoning.

### Gap Score Formula

$$gap\_score(T) = \alpha \times prerequisite\_gap(T) + \beta \times evidence\_weakness(T) + \gamma \times scope\_weight(T)$$

### So sánh với Knowledge Tracing

|                  | BKT        | DKT           | **KGA (ATIA)**            |
| ---------------- | ---------- | ------------- | ------------------------- |
| Biểu diễn        | P(mastery) | Hidden vector | Level + confidence        |
| Prerequisites    | Không      | Implicit      | **Explicit graph**        |
| Training data    | Cần        | Cần           | **Không cần (zero-shot)** |
| Interpretability | Cao        | Thấp          | **Cao**                   |

### Surface Knowledge Detection

Nếu topic level ≥ INTERMEDIATE nhưng prerequisite = NONE → Flag SURFACE_ONLY

---

## Agent 3: Question Strategy Agent (QSA)

**Mục đích:** Chọn hành động phỏng vấn tiếp theo dựa trên policy hình thức hóa.

<div class="columns">
<div>

### State Space

```
S = (top_gap, gap_count, phase,
     turn_ratio, coverage_mean,
     misconception_count,
     last_response_type,
     consecutive_non_substantive,
     clarification_count_current)
```

### Action Space

```
A = { PROBE_GAP, VERIFY_CLAIM,
      DEEPEN, PIVOT, REPHRASE,
      REDIRECT, CONCLUDE }
```

</div>
<div>

### Edge Case Rules (ưu tiên cao nhất)

- **CLARIFICATION** < 2 lần → REPHRASE
- **CLARIFICATION** ≥ 2 lần → PIVOT
- **OFF_TOPIC** < 2 lần → REDIRECT
- **OFF_TOPIC** ≥ 2 lần → PIVOT
- **DONT_KNOW** ≥ 2 lần → confirmed gap → PIVOT
- **Non-substantive** ≥ 3 liên tiếp → CONCLUDE

</div>
</div>

---

## QSA — POMDP Approximation

Bài toán phỏng vấn thích ứng được mô hình hóa như **POMDP**:

$$POMDP = (S, A, T, R, \Omega, O, \gamma)$$

| Thành phần | Ý nghĩa                                     |
| ---------- | ------------------------------------------- |
| $S$        | Trạng thái kiến thức thực của ứng viên (ẩn) |
| $A$        | 7 hành động phỏng vấn                       |
| $\Omega$   | Phản hồi ứng viên (tín hiệu nhiễu)          |
| $R$        | Information gain về trạng thái thực         |

**Giải chính xác → intractable.** QSA dùng **handcrafted approximation:**

- Duy trì **belief state** qua candidate_model (RA + KGA cập nhật)
- Chọn action qua **domain-informed heuristics**
- **Interpretable** + **ablation-friendly** + không cần training data

---

## Agent 4: Grounded Question Generator (GQG)

**Mục đích:** Chuyển quyết định chiến lược của QSA thành câu hỏi có grounding.

### Hành vi theo Action Type

| Action                     | GQG làm gì                                              |
| -------------------------- | ------------------------------------------------------- |
| PROBE_GAP / DEEPEN / PIVOT | Retrieve content → tạo câu hỏi mới → kiểm tra trùng lặp |
| VERIFY_CLAIM               | Tạo câu hỏi kiểm chứng misconception                    |
| REPHRASE                   | Cùng nội dung, diễn đạt đơn giản/rõ hơn                 |
| REDIRECT                   | Cùng topic, góc tiếp cận khác                           |

### Content-based Retrieval

- Retrieve từ Technical Content Corpus
- Đảm bảo factual accuracy cho câu hỏi
- Đính kèm reference chunks cho RA sử dụng ở turn sau

---

## Coordination Protocol — Per-turn Flow

```
Turn N: candidate_response received
       │
       ▼
  ┌─────────┐
  │  RA     │──→ Classify response type
  │ Step 0  │
  └────┬────┘
       │
       ├── SUBSTANTIVE/PARTIAL ──→ RA(full) → KGA → QSA → GQG → Câu hỏi tiếp
       │
       ├── DONT_KNOW ───────────→ RA(mark gap) → KGA → QSA → GQG → Câu hỏi tiếp
       │
       ├── CLARIFICATION ───────→ RA(classify only) ──────→ GQG(rephrase)
       │
       └── OFF_TOPIC ───────────→ RA(classify only) → QSA(redirect) → GQG
```

**Turn 0:** Khởi tạo ICO → QSA chọn entry topic (BASIC) → GQG tạo câu hỏi mở
**Turn cuối:** QSA → CONCLUDE → Feedback Synthesis → Assessment Report

---

## Knowledge Base — 3 thành phần

| Thành phần                        | Mục đích                     | Kích thước             | Agent sử dụng |
| --------------------------------- | ---------------------------- | ---------------------- | ------------- |
| **Prerequisite Dependency Graph** | Đồ thị phụ thuộc kiến thức   | ~100 nodes, ~250 edges | KGA           |
| **Technical Reference Corpus**    | Grounding cho đánh giá       | ~600-800 chunks        | RA            |
| **Question Pattern Bank**         | Mẫu câu hỏi theo action type | ~200-300 patterns      | QSA, GQG      |

### 3 Domains

| Domain                  | Đặc điểm graph                | Loại câu hỏi                    |
| ----------------------- | ----------------------------- | ------------------------------- |
| **DSA**                 | Sâu — chuỗi prerequisites dài | Conceptual, complexity analysis |
| **System Design**       | Rộng — nhiều cross-links      | Open-ended, trade-offs          |
| **Backend Engineering** | Module — clusters             | Practical, debugging            |

---

## Evaluation — Benchmark Construction

### 30 Candidate Profiles (synthetic)

| Level     | Số lượng | Mô tả                                          |
| --------- | -------- | ---------------------------------------------- |
| Junior    | 10       | Biết cơ bản, thiếu advanced, có misconceptions |
| Mid-level | 10       | Vững fundamentals, thiếu system design         |
| Senior    | 10       | Kiến thức sâu, misconceptions tinh tế          |

### Behavioral Variations

| Hành vi               | Số lượng | Test điều gì                                  |
| --------------------- | -------- | --------------------------------------------- |
| `standard`            | 18       | Core assessment accuracy                      |
| `evasive`             | 4        | PARTIAL classification + true level detection |
| `clarification-heavy` | 4        | REPHRASE flow                                 |
| `off-topic-prone`     | 2        | REDIRECT/PIVOT logic                          |
| `dont-know-heavy`     | 2        | Confirmed gaps, tránh probing loops           |

---

## Evaluation — Ground Truth & Baselines

### Ground Truth

- Xác định **trực tiếp từ profile** → không cần human annotation
- Knowledge state = profile, Gaps = topics_unknown ∪ topics_partial

### 8 Baselines (4 Ablation + 4 Published SOTA)

**Ablation baselines (B1–B4):**

| Baseline                    | Mô tả                        | Kiểm soát                      |
| --------------------------- | ---------------------------- | ------------------------------ |
| **B1: Monolithic LLM**      | Single prompt, không RAG     | Giá trị toàn bộ ATIA           |
| **B2: Vanilla RAG**         | Single RAG pipeline          | Multi-agent + hybrid reasoning |
| **B3: Single-Agent RAG**    | 1 agent, đầy đủ corpora      | Agent decomposition            |
| **B4: Static-Sequence RAG** | ATIA nhưng QSA = fixed order | Adaptive strategy              |

**Published SOTA baselines (B5–B8):**

| Baseline                            | Mô tả                                            | Kiểm soát                           |
| ----------------------------------- | ------------------------------------------------ | ----------------------------------- |
| **B5: LLM-as-Judge (Zheng'24)**     | Judge-style evaluation + CoT scoring + RAG       | ATIA vs. SOTA evaluation method     |
| **B6: LM-Interview (Li, EMNLP'24)** | Knowledge-guided LLM interview, 3-stage pipeline | ATIA vs. SOTA interview system      |
| **B7: KT+RAG (TutorLLM-style)**     | BERT-based Knowledge Tracing + RAG               | Graph-based vs. neural KT           |
| **B8: CAT/IRT (classical)**         | 3PL IRT model + max Fisher information           | ATIA vs. classical adaptive testing |

**Controls:** Cùng LLM backbone, cùng corpus, cùng max_turns, cùng simulator
**Quy mô:** 30 profiles × 9 systems = **270 interviews**, ~4,050 Q&A pairs

---

## Metrics & Statistical Analysis

| Metric                   | Đo gì                  | Định nghĩa hình thức                       |
| ------------------------ | ---------------------- | ------------------------------------------ |
| **AAS**                  | Chất lượng đánh giá    | `(1/\|T\|) × Σ sim(pred, gt)` per topic    |
| **Gap Detection F1**     | Hiệu quả KGA           | Precision × Recall trên gaps               |
| **Hallucination Rate**   | Độ tin cậy             | `\|contradictions\| / \|claims\|`          |
| **Topic Coverage**       | Chiến lược QSA         | `\|probed\| / \|in_scope\|`                |
| **Latency**              | Khả thi triển khai     | Wall-clock seconds per turn                |
| **Simulator Compliance** | Tính hợp lệ evaluation | `\|compliant\| / \|total\|` (target ≥ 95%) |

### Statistical Analysis

- **Wilcoxon signed-rank test** (non-parametric, paired by profile)
- **Bonferroni correction** (α = 0.05/8 ≈ 0.00625) — 8 baselines
- **Cliff's delta** (effect size) + **95% bootstrap CI**
- Tất cả metrics có **formal mathematical definitions** trong paper

---

## Ablation Study

| Variant         | Bỏ thành phần                        | Giả thuyết                            |
| --------------- | ------------------------------------ | ------------------------------------- |
| **ATIA-full**   | Không (hệ thống đầy đủ)              | —                                     |
| **ATIA−RAG**    | Bỏ toàn bộ retrieval                 | Hallucination tăng, AAS giảm          |
| **ATIA−KGA**    | KGA → LLM-only gap detection         | Gap F1 giảm, mất surface detection    |
| **ATIA−Policy** | QSA → random action selection        | Coverage giảm, depth progression giảm |
| **ATIA−Diff**   | Tất cả agents dùng chung 1 retrieval | Grounding rate giảm                   |

**Quy mô:** 30 profiles × 5 variants = 150 interviews bổ sung

---

## Expert Validation Study

### Thiết kế

- **3 senior engineers** (5+ năm kinh nghiệm) đánh giá độc lập
- **30 transcripts** (10/level × 2 systems: ATIA + best baseline)

### 4 nhiệm vụ đánh giá

| Nhiệm vụ              | Metric                        | Thang đo           |
| --------------------- | ----------------------------- | ------------------ |
| Chất lượng assessment | Expert Assessment Score (EAS) | Likert 1-5         |
| Độ thực tế simulator  | Simulator Realism Score (SRS) | Likert 1-5         |
| Chất lượng câu hỏi    | Question Quality Score (QQS)  | Likert 1-5         |
| Xác định gaps         | Expert-identified gaps        | So sánh với system |

### Phân tích

- Inter-rater: **Krippendorff's alpha** (target ≥ 0.7)
- AAS vs EAS: **Spearman's rho** (validate automated metric)
- Gaps: **Cohen's kappa**

---

## Giải quyết Circular Evaluation Bias

### Vấn đề

LLM (Claude) tạo ứng viên giả lập → LLM (Claude) đánh giá → kết quả có thể bị thổi phồng

### 3 lớp bảo vệ

| Lớp               | Giải pháp                   | Chi tiết                                                                                                       |
| ----------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Automated**     | Simulator Compliance Checks | Profile leakage detection + knowledge boundary verification bằng **cross-model** (GPT-4o-mini kiểm tra Claude) |
| **Expert**        | Expert Validation Study     | 3 engineers đánh giá SRS (simulator realism) + so sánh gaps                                                    |
| **Architectural** | Deterministic Ground Truth  | Ground truth = profile gốc, không phụ thuộc LLM judgment                                                       |

### Simulator Compliance Validation

```
Mỗi response được kiểm tra:
1. Profile Leakage: không lộ system prompt
2. Knowledge Boundary: đúng level theo profile
3. Behavioral Consistency: đúng style (evasive, standard...)
Target: Compliance Rate ≥ 95%
```

---

## Multi-Backbone Generalizability

### Mục tiêu

Chứng minh đóng góp kiến trúc không phụ thuộc vào LLM backbone cụ thể.

### Setup

- Chạy ATIA + 8 baselines trên **GPT-4o** với **10 profiles** (90 interviews)
- So sánh với kết quả Claude Sonnet

### Phân tích

| So sánh             | Metric                                      |
| ------------------- | ------------------------------------------- |
| Hiệu suất tuyệt đối | AAS, Gap F1, Hallucination Rate trên GPT-4o |
| Thứ hạng hệ thống   | Ranking có giữ nguyên không?                |
| Biên độ cải thiện   | ATIA margin nhất quán?                      |

- **Kendall's tau** ≥ 0.8 → kết quả backbone-consistent

---

## Timeline — 12 tuần

```
Tuần 1-3: Foundation          Tuần 3-6: Agent Dev
├─ KB construction             ├─ RA + KGA implementation
├─ Graph + Corpus + Patterns   ├─ QSA + GQG implementation
└─ ICO + interfaces            └─ Integration testing

Tuần 6-10: Evaluation          Tuần 10-12: Writing
├─ Candidate profiles          ├─ Sections 1-6
├─ Baselines                   ├─ Discussion + Conclusion
├─ Main run (150 interviews)   ├─ Internal review
├─ Ablation (150 interviews)   └─ Format + Submit
├─ Multi-backbone (50)
├─ Expert validation
└─ Statistical analysis
```

---

## Budget

| Hạng mục                                        | Chi phí ước tính |
| ----------------------------------------------- | ---------------- |
| Main experiment (270 interviews, Claude Sonnet) | ~$170            |
| Ablation (5 variants × 30 profiles)             | ~$95             |
| B7 KT model training (local GPU)                | ~$0              |
| B8 IRT item calibration (expert)                | ~$50             |
| Simulator compliance (GPT-4o-mini)              | ~$15             |
| KB construction + testing                       | ~$30             |
| Multi-backbone (90 interviews, GPT-4o)          | ~$80–120         |
| Expert validation (3 evaluators)                | ~$200–300        |
| Buffer                                          | ~$60             |
| **Tổng**                                        | **~$700–840**    |

### Chiến lược tiết kiệm (~$200-300)

- Claude Haiku cho Candidate Simulator → giảm ~$150-200
- Volunteer evaluators từ academic network → giảm ~$100-200
- B6/B7/B8 dùng chung simulator infrastructure
- **Budget thực tế: ~$550–700**

---

## Quản lý rủi ro

| Rủi ro                        | Mức độ | Giải pháp                                                           |
| ----------------------------- | ------ | ------------------------------------------------------------------- |
| **Circular evaluation bias**  | HIGH   | Cross-model compliance check + expert validation + deterministic GT |
| Simulator không thực tế       | HIGH   | Compliance ≥ 95% + SRS from experts; pilot 5 interviews             |
| Vượt budget                   | MEDIUM | Haiku cho simulator; ablation trên subset                           |
| Latency quá cao cho real-time | MEDIUM | Đo + báo cáo; thảo luận parallelization strategies                  |
| Multi-agent không hiệu quả    | HIGH   | Báo cáo trung thực; pivot sang hybrid-reasoning                     |
| Reviewer yêu cầu user study   | MEDIUM | Expert validation + deterministic GT; user study = future work      |
| Kết quả không generalize      | MEDIUM | Multi-backbone experiment trên GPT-4o (60 interviews)               |

---

## Positioning — So sánh với các hướng liên quan

<div class="columns">
<div>

### Khác ITS/CAT

- CAT: item selection từ item bank có sẵn
- **ATIA: generate câu hỏi mới**, grounded trong technical content
- CAT cần IRT calibration → ATIA zero-shot

### Khác Knowledge Tracing

- BKT/DKT: cần training data
- **ATIA: zero-shot** qua prerequisite graph + LLM
- Explicit gap_score decomposition

</div>
<div>

### Khác Standard RAG

- Single pipeline → **differentiated retrieval**
- Mỗi agent có retrieval strategy riêng

### Khác Multi-Agent LLM

- AutoGen/MetaGPT: general-purpose
- **ATIA: domain-specific** với formalized policy
- POMDP-grounded decision making

</div>
</div>

---

## Tóm tắt

<div class="highlight">

**ATIA** = Multi-agent RAG + Hybrid Knowledge-Gap Reasoning + Formalized Policy + Differentiated Retrieval

</div>

### 4 đóng góp chính

1. Kiến trúc multi-agent với retrieval strategy chuyên biệt cho từng agent
2. Hybrid reasoning (symbolic graph + neural) cho chẩn đoán kiến thức zero-shot
3. Assessment policy hình thức hóa (POMDP approximation) — interpretable & ablation-friendly
4. Framework đánh giá toàn diện: simulated benchmark + expert validation + multi-backbone

### Evaluation

- 270 simulated interviews × 9 systems (4 ablation + 4 published SOTA) + 150 ablation + 90 multi-backbone
- Simulator compliance validation (cross-model, ≥ 95%)
- Expert validation với 3 evaluators trên 30 transcripts
- Statistical rigor: Wilcoxon + Bonferroni + Cliff's delta + formal metric definitions
- Latency analysis cho deployment feasibility

**Target:** Expert Systems with Applications (Q1, IF 10.48) — Tháng 6/2026
