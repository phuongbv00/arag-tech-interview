# Research Proposal: Agentic RAG for Adaptive Technical Interview Assessment

**Target Journal:** Expert Systems with Applications (Elsevier, Q1, IF 10.48)
**Timeline:** 3 months (submission target: June 2026)
**Budget:** $100–500 (API calls)

---

## 1. Title (Working)

**"ATIA: An Agentic RAG Architecture with Hybrid Knowledge-Gap Reasoning for Adaptive Technical Interview Assessment"**

### Title rationale

- **ATIA** — memorable acronym, easy to reference
- "Agentic RAG" — positions within trending research direction, immediately signals technical depth
- "Hybrid Knowledge-Gap Reasoning" — signals the symbolic + neural contribution (not just prompt engineering)
- "Adaptive Technical Interview Assessment" — clear application domain

---

## 2. Abstract (Draft — ~250 words)

Automated technical interview systems powered by Large Language Models (LLMs) suffer from two compounding limitations: static question generation that fails to adapt to a candidate's evolving knowledge state, and response evaluation grounded solely in LLM parametric knowledge — a documented source of hallucination in high-stakes assessment contexts. This paper presents ATIA, a multi-agent Retrieval-Augmented Generation architecture for adaptive technical interview assessment. ATIA introduces three key mechanisms: (1) a hybrid knowledge-gap reasoning module that combines prerequisite dependency graph traversal with LLM-based response interpretation to diagnose candidate knowledge states, (2) a formalized assessment policy that maps candidate states to interviewing actions through an explicit decision function with defined state and action spaces, and (3) differentiated retrieval strategies across agents — assertion-based retrieval for response grounding, graph-based retrieval for gap diagnosis, and content-based retrieval for question generation — eliminating the single-retrieval-pipeline bottleneck of standard RAG. The system comprises four specialized agents (Response Analyst, Knowledge Gap Analyzer, Question Strategist, and Grounded Question Generator) coordinated through a shared Interview Context Object that provides full observability and traceability. We evaluate ATIA on a constructed benchmark of 150 simulated technical interviews spanning three software engineering domains (Data Structures & Algorithms, System Design, Backend Engineering), comparing against four baselines: monolithic LLM, vanilla RAG, single-agent RAG, and static-sequence RAG. Results demonstrate that ATIA achieves [X]% improvement in assessment alignment with expert ground truth, [X]% reduction in hallucinated evaluations, and [X]% better topic coverage compared to the strongest baseline. Ablation studies confirm that each architectural component contributes independently to system performance.

---

## 3. Research Questions

| ID      | Question                                                                                                                                                   | Measured by                                                                            |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **RQ1** | Does multi-agent decomposition improve assessment quality compared to monolithic LLM and single-agent RAG approaches?                                      | Assessment alignment score, hallucination rate                                         |
| **RQ2** | Does hybrid knowledge-gap reasoning (symbolic graph + neural interpretation) outperform purely neural gap detection?                                       | Gap detection precision/recall vs. expert-annotated gaps                               |
| **RQ3** | Does the formalized assessment policy produce better interview trajectories than static or random question sequences?                                      | Topic coverage, depth progression, assessment discrimination                           |
| **RQ4** | Do differentiated retrieval strategies across agents improve grounding quality compared to shared single-pipeline RAG?                                     | Grounding rate, factual accuracy of generated questions and evaluations                |
| **RQ5** | Does ATIA maintain assessment quality when candidates exhibit non-standard response behaviors (evasive, clarification-heavy, off-topic, don't-know-heavy)? | Edge Case Handling Rate, AAS comparison across behavioral profiles, Probing Efficiency |

---

## 4. Proposed Architecture — ATIA

### 4.1 Overview

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
    │ Response     │──▶│ Knowledge   │─▶│Question│──▶│Grounded  │
    │ Analyst (RA) │   │ Gap Analyzer│  │Strategy│   │Question  │
    │              │   │ (KGA)       │  │(QSA)   │   │Gen (GQG) │
    └──────────────┘   └─────────────┘  └────────┘   └──────────┘
     Assertion-based    Graph-based      Pattern-      Content-based
     retrieval          retrieval        based         retrieval
         │                  │            retrieval          │
         ▼                  ▼               │               ▼
    ┌─────────┐     ┌────────────┐          ▼         ┌──────────┐
    │Technical│     │Prerequisite│    ┌──────────┐    │Technical │
    │Reference│     │Dependency  │    │Question  │    │Content   │
    │Corpus   │     │Graph       │    │Pattern   │    │Corpus    │
    └─────────┘     └────────────┘    │Bank      │    └──────────┘
                                      └──────────┘
```

### 4.2 Interview Context Object (ICO) — Shared State

The ICO is the formal coordination mechanism. All agents read from and write to specific fields, enforcing specialization boundaries.

```
ICO = {
  candidate_model: {
    demonstrated: Map<TopicID → {
      level: NONE | SURFACE | INTERMEDIATE | DEEP,
      evidence: List<{turn, claim, grounding_result}>,
      confidence: float  // based on evidence count + consistency
    }>,
    gaps: PriorityQueue<{
      topic: TopicID,
      severity: float,       // f(dependency_depth, interview_scope_weight)
      gap_type: MISSING | SURFACE_ONLY | MISCONCEPTION,
      probing_count: int
    }>,
    misconceptions: List<{
      claim: string,
      correction: string,
      source_ref: string,
      turn_detected: int
    }>
  },

  interview_state: {
    phase: WARMUP | PROBING | DEEP_DIVE | SYNTHESIS,
    turn_count: int,
    max_turns: int,          // configurable (e.g., 15-20 for ~30min interview)
    questions_asked: List<{
      turn: int,
      question: string,
      topic: TopicID,
      difficulty: BASIC | INTERMEDIATE | ADVANCED,
      action_type: ActionType,
      response_type: SUBSTANTIVE | CLARIFICATION | DONT_KNOW | OFF_TOPIC | PARTIAL,
      response_summary: string
    }>,
    coverage: Map<TopicID → float>,  // 0.0 to 1.0
    consecutive_non_substantive: int, // resets on SUBSTANTIVE response
    clarification_count_current: int, // for current question, resets on new question
    confirmed_gaps: Set<TopicID>      // gaps confirmed by repeated DONT_KNOW, stop probing
  },

  knowledge_context: {
    grounding_cache: Map<string → {
      supported: bool,
      evidence: string,
      confidence: float
    }>,
    prerequisite_graph: DirectedGraph<TopicID, DependencyType>
  }
}
```

### 4.3 Agent Specifications

#### Agent 1: Response Analyst (RA)

**Purpose:** Classify candidate responses and evaluate substantive content through claim-level factual grounding.

**Unique retrieval:** Assertion-based retrieval from Technical Reference Corpus. Each technical claim extracted from candidate response → converted to verification query → retrieve reference passages → classify claim.

| Aspect           | Specification                                                                                                                                                                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input            | `candidate_response: string`, `current_question: string`, `ICO.knowledge_context.grounding_cache`                                                                                                                                                                                                 |
| Output           | Writes to `ICO.candidate_model.demonstrated`, `ICO.candidate_model.misconceptions`, `ICO.interview_state.response_type`                                                                                                                                                                           |
| Core process     | **(0) Response Classification** → (1) Extract technical claims via structured prompting → (2) For each claim: retrieve reference chunks (top-k=5, cross-encoder reranking) → (3) Classify: SUPPORTED / CONTRADICTED / PARTIALLY_CORRECT / UNVERIFIABLE → (4) Update candidate model with evidence |
| Retrieval corpus | Curated technical reference: official documentation, verified Q&A pairs, textbook excerpts                                                                                                                                                                                                        |
| Retrieval method | Dense embedding (voyage-3-lite) + cross-encoder reranker for claim verification                                                                                                                                                                                                                   |

**Step 0 — Response Classification (runs before claim extraction):**

RA first classifies the response into one of five types, which determines the downstream processing flow:

```
ResponseType = classify(candidate_response, current_question):

  SUBSTANTIVE     — Contains technical claims addressable to the question.
                    → Proceed with full claim extraction pipeline (Steps 1–4).
                    → Reset consecutive_non_substantive = 0.

  CLARIFICATION   — Candidate asks for clarification ("Do you mean X or Y?",
                    "Could you rephrase?", "What do you mean by Z?").
                    → Do NOT update candidate_model (no knowledge signal).
                    → Increment clarification_count_current.
                    → Signal: REPHRASE (skip KGA/QSA, route to GQG to rephrase).

  DONT_KNOW       — Candidate explicitly states lack of knowledge ("I don't know",
                    "I'm not familiar with that", "I haven't worked with X").
                    → Update candidate_model: set topic level = NONE, confidence = HIGH.
                    → Increment probing_count for topic.
                    → Signal: CONFIRMED_GAP (KGA will mark, QSA will decide next action).

  OFF_TOPIC       — Response has low semantic relevance to the question topic
                    (e.g., answering about REST APIs when asked about B-trees).
                    → Do NOT update candidate_model (no valid knowledge signal).
                    → Increment consecutive_non_substantive.
                    → Signal: REDIRECT (QSA will re-ask or pivot).

  PARTIAL         — Contains some relevant content but incomplete or hedged
                    ("I think it might be... but I'm not sure about...").
                    → Extract available claims (may be fewer).
                    → Update candidate_model with level = SURFACE, confidence = LOW.
                    → Proceed with reduced claim set.
```

**Classification method:** Single LLM call with structured output. Prompt includes current question for relevance assessment. Uses semantic similarity between response and question topic as additional signal for OFF_TOPIC detection (threshold: cosine < 0.3).

**Modified per-turn flow based on ResponseType:**

```
SUBSTANTIVE → RA(full) → KGA → QSA → GQG          // normal flow
PARTIAL     → RA(partial) → KGA → QSA → GQG        // normal flow, less data
DONT_KNOW   → RA(mark gap) → KGA → QSA → GQG       // fast update, QSA decides
CLARIFICATION → RA(classify only) → GQG(rephrase)   // skip KGA/QSA
OFF_TOPIC   → RA(classify only) → QSA(redirect)→GQG // skip KGA
```

**Key design decision:** RA does NOT produce a holistic score. It produces **structured, evidence-linked claim evaluations**. This enables traceability and ablation (compare RA-grounded scoring vs. LLM-parametric scoring).

#### Agent 2: Knowledge Gap Analyzer (KGA)

**Purpose:** Diagnose candidate knowledge state using hybrid symbolic + neural reasoning.

**Unique retrieval:** Graph-based retrieval from Prerequisite Dependency Graph.

| Aspect       | Specification                                                                                                                                                                                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Input        | `ICO.candidate_model.demonstrated`, `ICO.knowledge_context.prerequisite_graph`                                                                                                                                                                                                       |
| Output       | Writes to `ICO.candidate_model.gaps`, `ICO.interview_state.coverage`                                                                                                                                                                                                                 |
| Core process | (1) Traverse prerequisite graph starting from interview scope topics → (2) For each topic: compute gap_score based on prerequisite coverage → (3) Detect "surface knowledge" pattern (topic demonstrated but prerequisites missing) → (4) Prioritize gaps by severity × scope_weight |

**Hybrid reasoning formalization:**

```
gap_score(T) = α × prerequisite_gap(T) + β × evidence_weakness(T) + γ × scope_weight(T)

where:
  prerequisite_gap(T) = |missing_prerequisites(T)| / |all_prerequisites(T)|
  evidence_weakness(T) = 1 - avg(confidence of demonstrated sub-topics)
  scope_weight(T) = importance of T in interview scope (predefined)

  α + β + γ = 1 (tunable hyperparameters)
```

**Surface knowledge detection:**

```
If demonstrated[T].level ≥ INTERMEDIATE
   AND ∃ prerequisite P of T where demonstrated[P].level = NONE:
   → Flag T as SURFACE_ONLY, priority = HIGH
   → Rationale: candidate may have memorized T without understanding P
```

#### Agent 3: Question Strategy Agent (QSA)

**Purpose:** Select next interviewing action based on formalized policy.

**Unique retrieval:** Pattern-based retrieval from Question Pattern Bank (effective follow-up sequences for each action type).

| Aspect | Specification                                                      |
| ------ | ------------------------------------------------------------------ |
| Input  | `ICO.candidate_model.gaps`, `ICO.interview_state`                  |
| Output | `action: ActionType + parameters`, writes to `ICO.interview_state` |

**State space:**

```
S = (top_gap, gap_count, phase, turn_ratio, coverage_mean, misconception_count,
     last_response_type, consecutive_non_substantive, clarification_count_current)
  where turn_ratio = turn_count / max_turns
```

**Action space:**

```
A = {
  PROBE_GAP(topic, difficulty),
  VERIFY_CLAIM(misconception_id),
  DEEPEN(topic),
  PIVOT(new_topic),
  REPHRASE(original_question),     // rephrase current question with more context
  REDIRECT(topic),                  // re-ask about same topic, different angle
  CONCLUDE
}
```

**Policy π(S) → A (rule-based, deterministic):**

```
// ── Edge case rules (highest priority) ──────────────────────────

0a. IF last_response_type = CLARIFICATION AND clarification_count_current < 2:
      → REPHRASE(last_question)
      // Candidate asked for clarification — rephrase, don't penalize

0b. IF last_response_type = CLARIFICATION AND clarification_count_current ≥ 2:
      → PIVOT(next_highest_priority_topic)
      // Question is unclear despite rephrasing — move on

0c. IF last_response_type = OFF_TOPIC AND consecutive_non_substantive < 2:
      → REDIRECT(current_topic)
      // Try same topic from different angle

0d. IF last_response_type = OFF_TOPIC AND consecutive_non_substantive ≥ 2:
      → PIVOT(next_highest_priority_topic)
      // Candidate consistently off-topic — move on

0e. IF last_response_type = DONT_KNOW AND top_gap.probing_count ≥ 2:
      → Add topic to confirmed_gaps (stop probing this topic)
      → PIVOT(next_highest_priority_topic NOT in confirmed_gaps)
      // Confirmed gap — don't keep asking about what they don't know

0f. IF consecutive_non_substantive ≥ 3:
      → CONCLUDE
      // Candidate unable or unwilling to engage — end gracefully

// ── Standard rules (original policy) ────────────────────────────

1. IF misconception_count > 0 AND turn_ratio < 0.85:
     → VERIFY_CLAIM(most_recent_misconception)

2. ELIF top_gap.gap_type = SURFACE_ONLY AND top_gap.probing_count < 2:
     → PROBE_GAP(top_gap.prerequisite, difficulty=BASIC)
     // Probe the prerequisite they're missing, not the surface topic

3. ELIF top_gap.severity > threshold_high AND turn_ratio < 0.7:
     → PROBE_GAP(top_gap.topic, difficulty=adaptive)
     // adaptive = one level below current demonstrated level

4. ELIF coverage_mean < target_coverage AND turn_ratio > 0.5:
     → PIVOT(lowest_coverage_topic_in_scope)

5. ELIF ∃ topic with level = INTERMEDIATE AND not yet deepened:
     → DEEPEN(topic)

6. ELIF turn_ratio ≥ 0.9 OR (coverage_mean ≥ target AND gap_count = 0):
     → CONCLUDE

7. ELSE:
     → PROBE_GAP(top_gap.topic, difficulty=BASIC)  // default
```

**Edge case design rationale:**

- Rules 0a–0f have **highest priority** — edge cases must be handled before any strategic decision.
- `confirmed_gaps` prevents infinite probing loops: once a topic is confirmed as unknown (2+ DONT_KNOW responses), it is recorded but no longer actively probed.
- `consecutive_non_substantive ≥ 3` triggers early termination — if a candidate gives 3 consecutive non-substantive responses (any combination of OFF_TOPIC, CLARIFICATION, DONT_KNOW), the system concludes rather than continuing an unproductive loop.
- REPHRASE vs REDIRECT distinction: REPHRASE keeps the same question intent but changes wording; REDIRECT keeps the same topic but asks a different question. This avoids both repetition and topic abandonment.

**Why rule-based, not learned:** (a) Interpretability — every decision traceable to a rule; (b) No training data needed (critical for solo author / 3-month timeline); (c) Ablation-friendly — can replace with random/sequential/LLM-decided policies cleanly.

#### Agent 4: Grounded Question Generator (GQG)

**Purpose:** Transform QSA's strategic decision into a well-formed, grounded question.

**Unique retrieval:** Content-based retrieval from Technical Content Corpus — retrieve relevant technical material to ensure question accuracy.

| Aspect       | Specification                                                                                                                                                                                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input        | `action` from QSA, `ICO.interview_state.questions_asked`, `ICO.candidate_model`                                                                                                                                                                                                 |
| Output       | `question: string`, `reference_chunks: List<string>`                                                                                                                                                                                                                            |
| Core process | (1) Receive action (e.g., PROBE_GAP("hash_tables", INTERMEDIATE)) → (2) Retrieve relevant technical content → (3) Generate question grounded in retrieved content → (4) Verify no overlap with questions_asked → (5) Output question + attach reference chunks for later RA use |

**GQG behavior by action type:**

```
PROBE_GAP / DEEPEN / PIVOT / VERIFY_CLAIM:
  → Standard flow: retrieve content → generate new question → check overlap

REPHRASE(original_question):
  → Retrieve same topic content but generate simpler/clearer wording
  → Add contextual hints (e.g., "To put it another way..." or
    "Let me make this more specific...")
  → Do NOT ask a fundamentally different question

REDIRECT(topic):
  → Generate a different question on the SAME topic
  → Approach from a different angle (e.g., if first question was conceptual,
    try practical; if abstract, try example-based)
  → Must NOT overlap with questions_asked
```

### 4.4 Coordination Protocol

**Per-turn flow:** Sequential, deterministic, with conditional branching based on response type:

```
Turn N (N ≥ 1):
  candidate_response received
       │
       ▼
  ┌─────────┐
  │  RA     │──→ Classify response type
  │ Step 0  │
  └────┬────┘
       │
       ├── SUBSTANTIVE/PARTIAL ──→ RA(full) → KGA → QSA → GQG → Next Question
       │
       ├── DONT_KNOW ───────────→ RA(mark gap) → KGA → QSA → GQG → Next Question
       │
       ├── CLARIFICATION ───────→ RA(classify only) ──────→ GQG(rephrase) → Rephrased Question
       │                          (skip KGA, QSA decides via edge case rule 0a/0b)
       │
       └── OFF_TOPIC ───────────→ RA(classify only) → QSA(redirect/pivot) → GQG → Next Question
                                  (skip KGA — no valid data to analyze)
```

**Rationale for conditional branching:** Non-substantive responses contain no valid knowledge signal for KGA to analyze. Processing them through the full pipeline wastes tokens and may introduce noise into the candidate model. The branching ensures each response type receives appropriate treatment.

### 4.5 Turn 0 — Interview Initialization Protocol

Turn 0 is distinct from subsequent turns: no candidate response exists yet, so RA and KGA are skipped.

```
Turn 0 Protocol:

  Input: interview_scope = {
    target_role: string,              // e.g., "Backend Engineer, Mid-level"
    domains: List<DomainID>,          // e.g., [DSA, SD, BE]
    priority_topics: List<TopicID>,   // optional: topics the interviewer cares about
    max_turns: int,                   // e.g., 15
    target_coverage: float            // e.g., 0.6 (probe ≥60% of in-scope topics)
  }

  ICO Initialization:
    - candidate_model: all topics set to level=NONE, confidence=0
    - interview_state.phase = WARMUP
    - coverage: all topics = 0.0
    - prerequisite_graph: loaded from KB
    - confirmed_gaps: empty
    - consecutive_non_substantive: 0

  QSA Turn-0 Policy:
    1. Identify entry_topic:
       - If priority_topics specified → select first priority topic
       - Else → select topic with highest scope_weight in primary domain
    2. Select difficulty = BASIC
       // Always start easy: calibrates candidate level from first response
    3. Action = PROBE_GAP(entry_topic, BASIC)

  GQG Turn-0 Behavior:
    - Generate a broad, open-ended question on entry_topic
    - Question should allow candidate to demonstrate range
      (not a narrow factual recall question)
    - Example: "Can you walk me through how a hash map works
      under the hood — how it stores data, handles collisions,
      and what trade-offs different strategies offer?"
    - Grounded in retrieved technical content (factual accuracy ensured)
```

**Final turn:** QSA outputs CONCLUDE → Feedback Synthesis runs once, reading full ICO to produce assessment report.

---

## 5. Technical Domains & Knowledge Base Construction

### 5.1 Domain Selection Rationale

Three domains chosen for complementary properties:

| Domain                                 | Why chosen                                                                                                            | Prerequisite graph complexity                          | Question types                                             |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| **Data Structures & Algorithms (DSA)** | Deep prerequisite chains (e.g., balanced BST → BST → binary tree → tree → graph). Well-structured taxonomy available. | HIGH — deep dependency trees                           | Conceptual, complexity analysis, trade-off reasoning       |
| **System Design (SD)**                 | Broad and lateral — fewer strict prerequisites but many interconnected concepts. Tests coverage strategy.             | MEDIUM — wide, shallow graph with cross-links          | Open-ended design, trade-off discussion, scaling reasoning |
| **Backend Engineering (BE)**           | Mix of conceptual + practical knowledge. Includes API design, database, caching, security.                            | MEDIUM — modular clusters with some cross-domain links | Practical application, debugging reasoning, best practices |

### 5.2 Knowledge Base Components (build from scratch)

**Component 1: Prerequisite Dependency Graph (for KGA)**

| Aspect              | Plan                                                                                                          |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| Source              | ACM CS2023 curriculum guidelines + manual curation from widely-used textbooks (CLRS for DSA, DDIA for SD)     |
| Size target         | ~80–120 topic nodes, ~200–300 dependency edges across 3 domains                                               |
| Format              | JSON adjacency list with metadata (topic_id, name, domain, difficulty_tier, description)                      |
| Construction effort | ~2–3 days manual work; validate by checking 10 random paths make logical sense                                |
| Example             | `hash_table → array, hash_function, collision_resolution`; `load_balancer → DNS, reverse_proxy, health_check` |

**Component 2: Technical Reference Corpus (for RA — grounding)**

| Aspect              | Plan                                                                                                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source              | Official documentation (Python docs, PostgreSQL docs, Redis docs), open-license textbook excerpts, curated Stack Overflow answers (CC-BY-SA), Wikipedia technical articles |
| Size target         | ~500–800 chunks (each ~300–500 tokens)                                                                                                                                     |
| Processing          | Chunk by semantic section → embed with voyage-3-lite → store in local vector DB (ChromaDB or Qdrant)                                                                       |
| Construction effort | ~3–4 days (scraping + chunking + embedding)                                                                                                                                |

**Component 3: Question Pattern Bank (for QSA/GQG)**

| Aspect              | Plan                                                                                                                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source              | Publicly available interview question collections (LeetCode discussion, System Design Primer, tech blog posts under CC license), manually categorized by topic + difficulty + action_type |
| Size target         | ~200–300 question patterns (not exact questions — patterns like "Explain the trade-off between X and Y in the context of Z")                                                              |
| Construction effort | ~2 days curation + categorization                                                                                                                                                         |

**Total KB construction effort: ~8–10 days**

### 5.3 Budget Estimation for KB Construction

| Task                                       | API cost    |
| ------------------------------------------ | ----------- |
| Embedding 800 chunks (voyage-3-lite)       | ~$2–5       |
| LLM-assisted question pattern extraction   | ~$10–15     |
| LLM-assisted prerequisite graph validation | ~$5–10      |
| **Subtotal KB**                            | **~$20–30** |

---

## 6. Evaluation Design

### 6.1 Benchmark Construction: Simulated Interview Corpus

Since user study is out of scope, we construct a **synthetic but rigorous evaluation benchmark**:

**Step 1: Create Candidate Profiles (synthetic)**

Design 30 candidate profiles across skill levels and behavioral patterns:

| Level     | Count | Description                                                           |
| --------- | ----- | --------------------------------------------------------------------- |
| Junior    | 10    | Knows basics, gaps in advanced topics, some misconceptions            |
| Mid-level | 10    | Solid fundamentals, gaps in system design / distributed systems       |
| Senior    | 10    | Deep knowledge with occasional subtle misconceptions, strong coverage |

**Behavioral variations** (distributed across levels to test edge case handling):

| Behavior tag          | Count | Applied to             | Description                                                            |
| --------------------- | ----- | ---------------------- | ---------------------------------------------------------------------- |
| `standard`            | 18    | 6 per level            | Answers directly; says "I don't know" when appropriate                 |
| `evasive`             | 4     | 2 Junior, 2 Mid        | Gives vague, hedging answers; frequently PARTIAL responses             |
| `clarification-heavy` | 4     | 1 per level + 1 Junior | Frequently asks "what do you mean?" before answering                   |
| `off-topic-prone`     | 2     | 1 Junior, 1 Mid        | Occasionally drifts to tangential topics                               |
| `dont-know-heavy`     | 2     | 2 Junior               | Says "I don't know" for most topics including some they partially know |

Each profile = JSON specifying:

```json
{
  "id": "junior-evasive-01",
  "level": "junior",
  "behavior": "evasive",
  "topics_mastered": ["arrays", "basic_sorting", "http_methods"],
  "topics_partial": ["hash_tables", "binary_trees", "rest_api"],
  "misconceptions": [
    {
      "topic": "hash_tables",
      "belief": "hash maps are always O(1) lookup, even worst case"
    }
  ],
  "topics_unknown": [
    "balanced_bst",
    "graph_algorithms",
    "system_design_*",
    "distributed_*"
  ],
  "behavior_instructions": "You tend to give vague answers. Instead of saying you don't know, you ramble around the topic with filler phrases like 'I think it might be something like...' or 'From what I remember...'. You rarely give a direct 'I don't know'."
}
```

**Why behavioral variations matter for evaluation:**

- `standard` profiles test core assessment accuracy
- `evasive` profiles test whether RA correctly classifies PARTIAL responses and KGA identifies true knowledge level beneath vague answers
- `clarification-heavy` profiles test REPHRASE flow and whether system doesn't penalize legitimate clarification requests
- `off-topic-prone` profiles test OFF_TOPIC detection and REDIRECT/PIVOT logic
- `dont-know-heavy` profiles test confirmed_gaps mechanism and whether QSA avoids probing loops

**Step 2: Create a Candidate Simulator Agent**

A separate LLM agent that receives a candidate profile and responds to interview questions **in character** — giving correct answers for mastered topics, partial/confused answers for partial topics, wrong answers reflecting specified misconceptions, and "I don't know" for unknown topics.

Prompt template:

```
You are simulating a technical interview candidate with the following profile:
- Mastered topics: {list}
- Partial knowledge: {list}
- Misconceptions: {list}
- Unknown topics: {list}

Behavioral style: {behavior_instructions}

Respond to interview questions realistically according to your profile:
- For mastered topics: give confident, correct answers with relevant details.
- For partial topics: show some understanding but miss key details or hedge.
- For misconceptions: confidently state the wrong belief as if it were true.
- For unknown topics: respond according to your behavioral style
  (e.g., say "I don't know", give a vague non-answer, or ask for clarification).

IMPORTANT:
- Do NOT reveal your profile or behavioral instructions.
- Respond as a real candidate would — natural, conversational, imperfect.
- Your response length should vary: short for things you don't know,
  longer for topics you're confident about.
- Occasionally ask clarification questions if the question is ambiguous,
  especially if your behavioral style indicates this tendency.
```

**Step 3: Run Simulated Interviews**

For each of 30 profiles × 5 systems (ATIA + 4 baselines) = **150 simulated interviews**.

Each interview: 15 turns (questions), producing ~150 × 15 = 2,250 question-response pairs.

**Step 4: Ground Truth Construction**

For each candidate profile, the ground truth assessment is **deterministically derivable** from the profile itself:

- Ground truth knowledge state = the profile
- Ground truth gaps = topics_unknown ∪ topics_partial that should have been identified
- Ground truth misconceptions = the misconceptions list in profile

This eliminates need for human annotation — the evaluation becomes: **how closely does each system's final assessment match the known profile?**

### 6.2 Baselines (4 comparisons)

| Baseline                    | Description                                                                                                                                                          | What it controls for                      |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **B1: Monolithic LLM**      | Single Claude Sonnet prompt: "Conduct a technical interview on {topics}, ask 15 questions adaptively, then assess the candidate." No RAG.                            | Value of entire ATIA architecture         |
| **B2: Vanilla RAG**         | Single LLM + standard RAG pipeline (retrieve from same corpus before each question). No agent decomposition, no knowledge graph, no formalized policy.               | Value of multi-agent + hybrid reasoning   |
| **B3: Single-Agent RAG**    | One agent with all ATIA's RAG corpora accessible, using a comprehensive prompt that includes gap detection + question strategy instructions. No agent decomposition. | Value of agent decomposition specifically |
| **B4: Static-Sequence RAG** | Same as ATIA but QSA replaced with fixed question sequence (pre-determined order of topics). All other agents intact.                                                | Value of adaptive question strategy       |

### 6.3 Metrics

| Metric                               | Definition                                                                                                                                                                                                               | Measures                           |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| **Assessment Alignment Score (AAS)** | Cosine similarity between system's final candidate_model and ground truth profile, computed per-topic.                                                                                                                   | Overall assessment quality         |
| **Gap Detection F1**                 | F1 score of identified gaps vs. ground truth gaps (topics_unknown + topics_partial).                                                                                                                                     | KGA effectiveness                  |
| **Misconception Detection Rate**     | Recall of detected misconceptions vs. ground truth misconceptions.                                                                                                                                                       | RA claim-grounding effectiveness   |
| **Hallucination Rate**               | % of system's assessment claims that contradict ground truth profile (e.g., marking a mastered topic as "gap" or vice versa).                                                                                            | Assessment reliability             |
| **Topic Coverage**                   | % of in-scope topics that were probed at least once during interview.                                                                                                                                                    | QSA strategy effectiveness         |
| **Depth Progression Score**          | Average difficulty increase across turns for topics probed multiple times. Measures if system adapts difficulty.                                                                                                         | Adaptive behavior quality          |
| **Grounding Rate**                   | % of RA's claim evaluations that include retrieved evidence (vs. parametric-only judgments).                                                                                                                             | RAG utilization effectiveness      |
| **Edge Case Handling Rate**          | % of non-substantive responses (CLARIFICATION, OFF_TOPIC, DONT_KNOW) correctly classified and appropriately handled (no candidate_model corruption from off-topic, no infinite probing loops, clarifications rephrased). | Response classification robustness |
| **Probing Efficiency**               | Ratio of substantive turns to total turns. Higher = less wasted turns on rephrasing/redirecting. Reported separately for standard vs. edge-case behavioral profiles.                                                     | Practical interview quality        |
| **Efficiency**                       | Total tokens consumed per interview (input + output).                                                                                                                                                                    | Practical cost considerations      |

### 6.4 Ablation Study Design

| Variant             | Removed component                                                                            | Hypothesis                                                    |
| ------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| ATIA-full           | None (complete system)                                                                       | —                                                             |
| ATIA−RAG            | All retrieval removed; agents use LLM parametric knowledge only                              | Hallucination rate increases, AAS decreases                   |
| ATIA−KGA            | KGA replaced with LLM-only gap detection (no prerequisite graph)                             | Gap Detection F1 decreases, surface-knowledge detection drops |
| ATIA−Policy         | QSA policy replaced with random action selection (still uses action types but random choice) | Coverage decreases, depth progression drops                   |
| ATIA−Differentiated | All agents share single retrieval pipeline (same corpus, same strategy)                      | Grounding rate decreases, question quality drops              |

### 6.5 Budget Estimation for Experiments

| Task                                              | Calculation                | Estimated cost |
| ------------------------------------------------- | -------------------------- | -------------- |
| 150 interviews × 15 turns × ~2 agent calls/turn   | ~4,500 Claude Sonnet calls |                |
| Average ~2,000 tokens/call (input+output)         | ~9M tokens total           |                |
| Claude Sonnet input ($3/MTok) + output ($15/MTok) | ~$27 input + ~$67 output   | ~$95           |
| Ablation (5 variants × 150 interviews)            | 5× above                   | ~$475          |
| KB construction + testing                         |                            | ~$30           |
| Buffer for debugging, reruns                      |                            | ~$50           |
| **TOTAL**                                         |                            | **~$450–550**  |

> ⚠️ **Budget is tight.** Mitigation strategies:
>
> - Run ablations on subset first (10 profiles = 50 interviews) to validate setup before full run
> - Use Claude Haiku for Candidate Simulator (cheaper, sufficient for simulating responses)
> - Cache retrieval results aggressively (same topic queries across interviews)
> - Estimated budget with Haiku for simulator: reduces to ~$300–350

---

## 7. Contribution Statement (for ESWA)

This paper makes four contributions:

1. **Multi-agent architecture with differentiated retrieval** — Unlike standard RAG systems that use a single retrieval pipeline, ATIA assigns each agent a specialized retrieval strategy matched to its function (assertion-based for grounding, graph-based for gap analysis, content-based for question generation), enabling more precise information access.

2. **Hybrid knowledge-gap reasoning** — ATIA combines symbolic prerequisite graph traversal with neural response interpretation for knowledge diagnosis, addressing the documented unreliability of purely LLM-based knowledge assessment. The "surface knowledge" detection pattern (topic demonstrated but prerequisites missing) is a novel diagnostic capability.

3. **Formalized adaptive assessment policy** — The question strategy agent operates on an explicit state-action policy with defined state space, action space, and deterministic rules, enabling full interpretability and clean ablation — a departure from opaque end-to-end LLM decision-making.

4. **Comprehensive evaluation framework** — Profile-based simulated interview benchmark that enables large-scale automated evaluation with deterministic ground truth, providing reproducible comparison across systems without requiring human subjects.

---

## 8. Timeline (12 weeks)

### Phase 1: Foundation (Weeks 1–3)

| Week | Task                                            | Deliverable                                          |
| ---- | ----------------------------------------------- | ---------------------------------------------------- |
| 1    | Build prerequisite dependency graph (3 domains) | `prerequisite_graph.json` (~100 nodes, ~250 edges)   |
| 1–2  | Curate & chunk Technical Reference Corpus       | ~600–800 chunks embedded in vector DB                |
| 2    | Build Question Pattern Bank                     | ~200 categorized question patterns                   |
| 3    | Implement ICO data structure + agent interfaces | Working LangGraph scaffold with ICO state management |

### Phase 2: Agent Development (Weeks 3–6)

| Week | Task                                                                                   | Deliverable                                               |
| ---- | -------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 3–4  | Implement RA (claim extraction + assertion-based retrieval + grounding classification) | RA agent with unit tests on 20 sample responses           |
| 4    | Implement KGA (graph traversal + gap scoring + surface knowledge detection)            | KGA agent with unit tests on 10 sample candidate states   |
| 5    | Implement QSA (policy rules + action selection)                                        | QSA agent with unit tests covering all 5 action types     |
| 5–6  | Implement GQG (action → grounded question generation)                                  | GQG agent with unit tests on all action types × 3 domains |
| 6    | Integration testing: full interview loop (10 sample runs)                              | End-to-end working system                                 |

### Phase 3: Evaluation (Weeks 6–9)

| Week | Task                                                             | Deliverable                                |
| ---- | ---------------------------------------------------------------- | ------------------------------------------ |
| 6–7  | Build 30 candidate profiles + Candidate Simulator                | Profile JSONs + simulator prompt validated |
| 7    | Implement 4 baselines                                            | All baselines runnable on same benchmark   |
| 7–8  | Pilot run: 10 profiles × 5 systems (50 interviews)               | Preliminary results, debug issues          |
| 8    | Full run: 30 profiles × 5 systems (150 interviews)               | Main comparison results                    |
| 8–9  | Ablation run: 30 profiles × 5 ablation variants (150 interviews) | Ablation results                           |
| 9    | Compute all metrics, generate tables/figures                     | Complete results section data              |

### Phase 4: Writing & Submission (Weeks 9–12)

| Week  | Task                                                                         | Deliverable      |
| ----- | ---------------------------------------------------------------------------- | ---------------- |
| 9–10  | Write Sections 1–4 (Intro, Related Work, Problem Formulation, Architecture)  | ~12–15 pages     |
| 10–11 | Write Sections 5–6 (Experiments, Results & Analysis)                         | ~8–10 pages      |
| 11    | Write Discussion, Conclusion, Abstract                                       | ~3–4 pages       |
| 11–12 | Internal review cycle (self-review after 2-day break), polish figures/tables | Final manuscript |
| 12    | Format for ESWA, prepare supplementary materials, submit                     | Submission       |

### Risk Buffer

- Week 12 also serves as buffer for Phase 2–3 delays
- If budget runs out during ablation: run ablation on 15 profiles instead of 30 (still statistically meaningful)
- If one domain's KB quality is poor: report it as limitation, focus analysis on 2 stronger domains

---

## 9. Risk Assessment & Mitigation

| Risk                                                               | Severity | Likelihood | Mitigation                                                                                                                                                                                      |
| ------------------------------------------------------------------ | -------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Candidate Simulator produces unrealistic responses                 | HIGH     | MEDIUM     | Validate with 5 pilot interviews; tune prompt; add response quality checks                                                                                                                      |
| Claude API cost exceeds budget                                     | MEDIUM   | MEDIUM     | Use Haiku for simulator; cache retrieval; run pilot on subset first                                                                                                                             |
| Prerequisite graph quality too low for meaningful KGA contribution | HIGH     | LOW        | Validate graph with 3 external senior engineers (informal review, not co-authorship); cross-reference with ACM curriculum                                                                       |
| Ablation shows multi-agent doesn't help vs. single-agent           | HIGH     | MEDIUM     | If confirmed: honestly report, pivot contribution to hybrid-reasoning + differentiated-retrieval (which are testable independently)                                                             |
| ESWA desk-reject (scope mismatch)                                  | LOW      | LOW        | ESWA explicitly lists multi-agent systems, knowledge management, and education in scope                                                                                                         |
| Reviewer requests user study                                       | MEDIUM   | MEDIUM     | Acknowledge in Discussion as limitation; argue simulated benchmark with deterministic ground truth provides stronger internal validity than small user study; propose user study as future work |
| Solo author bias perception                                        | LOW      | MEDIUM     | Provide reproducibility package (code + data + configs); emphasize deterministic ground truth eliminates subjective annotation                                                                  |

---

## 10. Paper Outline with Page Budget (ESWA format, ~25–30 pages)

| Section                         | Pages   | Key content                                                                                                                                                                                                                                                                 |
| ------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Introduction**             | 2.5     | Problem, limitations of existing approaches, contribution summary (4 bullets), paper structure                                                                                                                                                                              |
| **2. Related Work**             | 3.5     | (2.1) RAG systems & Agentic RAG, (2.2) Multi-agent LLM architectures, (2.3) Automated interview/assessment systems, (2.4) Knowledge gap detection in education/assessment, (2.5) Gap statement positioning ATIA                                                             |
| **3. Problem Formulation**      | 2       | Formal definition: inputs, outputs, quality criteria, constraints                                                                                                                                                                                                           |
| **4. ATIA Architecture**        | 7       | (4.1) Overview + ICO, (4.2) RA with Response Classification, (4.3) KGA with formalization, (4.4) QSA with policy definition + edge case rules, (4.5) GQG, (4.6) Coordination protocol with conditional branching, (4.7) Turn 0 initialization, (4.8) Implementation details |
| **5. Experimental Setup**       | 4       | (5.1) Benchmark construction, (5.2) Candidate profiles, (5.3) Baselines, (5.4) Metrics, (5.5) Implementation details                                                                                                                                                        |
| **6. Results & Analysis**       | 5       | (6.1) Main comparison (Table), (6.2) Ablation study (Table), (6.3) Per-domain analysis, (6.4) Case study (1 detailed interview trace), (6.5) Efficiency analysis                                                                                                            |
| **7. Discussion**               | 2.5     | (7.1) Key findings interpretation, (7.2) Limitations, (7.3) Threats to validity, (7.4) Practical implications                                                                                                                                                               |
| **8. Conclusion & Future Work** | 1.5     | Summary, future directions (user study, learned policy, multi-modal)                                                                                                                                                                                                        |
| **References**                  | ~2      | 50–70 references                                                                                                                                                                                                                                                            |
| **TOTAL**                       | **~30** |                                                                                                                                                                                                                                                                             |

---

## 11. Key References to Cite (Preliminary)

### RAG & Agentic RAG

- Lewis et al. (2020) — Original RAG paper
- Gao et al. (2024) — RAG survey
- Singh et al. (2025) — Agentic RAG survey
- Agentic RAG architecture survey (arXiv:2501.09136)

### Multi-Agent LLM Systems

- Wu et al. (2023) — AutoGen
- Hong et al. (2023) — MetaGPT
- ESWA multi-agent papers (drone inspection, industrial operations)
- CIR3 multi-agent Q&A generation (Knowledge-Based Systems, 2025)

### Assessment & Interview

- LLM-as-judge literature (Zheng et al., 2024)
- Automated scoring papers from CAEAI (Latif & Zhai 2024; Lee et al. 2024)
- RAG-for-education survey on CAEAI (2025)

### Knowledge Gap Detection

- Bayesian Knowledge Tracing (Corbett & Anderson, 1995)
- Deep Knowledge Tracing (Piech et al., 2015)
- Prerequisite learning literature

### Evaluation Methodology

- Simulated user methodology in IR (Azzopardi & Zuccon, 2016)
- LLM-as-judge (Zheng et al., 2024)
- RAGAS evaluation framework (Es et al., 2024)

---

## 12. Reproducibility Plan

| Artifact                   | Status      | Plan                                                                           |
| -------------------------- | ----------- | ------------------------------------------------------------------------------ |
| Source code                | Will create | GitHub repo (public upon acceptance, or upon submission as supplementary)      |
| Prerequisite graph         | Will create | JSON file in repo                                                              |
| Technical reference corpus | Will create | Document sources + processing scripts (raw data may have license restrictions) |
| Candidate profiles         | Will create | 30 JSON profiles in repo                                                       |
| Experiment configs         | Will create | All hyperparameters, prompt templates, model versions                          |
| Raw results                | Will create | All 300 interview transcripts (150 main + 150 ablation) as JSON                |
| Analysis scripts           | Will create | Python notebooks reproducing all tables/figures                                |

---

## 13. Decision Log

| Decision            | Chosen               | Alternatives considered   | Rationale                                                                            |
| ------------------- | -------------------- | ------------------------- | ------------------------------------------------------------------------------------ |
| Target journal      | ESWA                 | CAEAI, IJAIED, KBS        | Technical framing fits ESWA best; no education theory needed; no user study required |
| LLM backbone        | Claude Sonnet        | GPT-4o, open-source       | Author familiarity; strong instruction following; cost-effective for budget          |
| Evaluation approach | Simulated interviews | User study, expert panel  | Solo author + 3-month timeline; deterministic ground truth stronger for ablation     |
| QSA policy          | Rule-based           | Learned (RL), LLM-decided | Interpretable, no training data needed, clean ablation possible                      |
| Number of domains   | 3                    | 1 (deeper), 5 (broader)   | 3 balances depth vs. generalizability claims for single-author effort                |
| Candidate profiles  | 30 (10 per level)    | 10, 50, 100               | 30 provides reasonable statistical power while fitting budget                        |

---

_Proposal version: 1.0_
_Created: March 2026_
_Target submission: June 2026_
