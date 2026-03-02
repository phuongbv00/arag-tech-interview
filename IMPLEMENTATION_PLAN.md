# Implementation Plan: Adaptive Technical Interviewing Using Agentic RAG

## 1. Project Structure

```
arag-tech-interview/
├── pyproject.toml
├── PROPOSAL.md
├── IMPLEMENTATION_PLAN.md
├── .env.example
├── .gitignore
├── configs/
│   ├── default.yaml              # Main config (models, thresholds, session params)
│   ├── agents.yaml               # Per-agent prompt templates and parameters
│   └── evaluation.yaml           # RAGAS metrics, persona definitions, ablation toggles
├── data/
│   └── kb/
│       ├── concepts.json         # 30-40 DSA concept entries (source of truth)
│       └── README.md             # KB authoring guidelines
├── src/
│   ├── __init__.py
│   ├── config.py             # Pydantic Settings loader (YAML + env vars)
│   ├── models.py             # Shared Pydantic data models
│   ├── kb/
│   │   ├── __init__.py
│   │   ├── schema.py         # ConceptEntry Pydantic model
│   │   ├── loader.py         # Load concepts.json → validate → embed → ChromaDB
│   │   └── retriever.py      # ChromaDB query helpers
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── dma.py            # Dialogue Manager Agent (hybrid)
│   │   ├── kgda.py           # Knowledge Gap Detection Agent
│   │   ├── qga.py            # Question Generation Agent
│   │   ├── raa.py            # Response Assessment Agent
│   │   └── fsa.py            # Feedback Synthesis Agent
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py          # LangGraph InterviewState TypedDict
│   │   ├── nodes.py          # Node functions wrapping each agent
│   │   └── builder.py        # StateGraph construction + compilation
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── dma_prompts.py
│   │   ├── kgda_prompts.py
│   │   ├── qga_prompts.py
│   │   ├── raa_prompts.py
│   │   └── fsa_prompts.py
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── personas.py       # L1/L3 persona profile builders
│   │   └── candidate.py      # Claude 3.5 Sonnet candidate simulator
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py        # Custom metrics (GapCoverageRate, etc.)
│   │   ├── ragas_adapter.py  # Format session data for RAGAS evaluate()
│   │   ├── runner.py         # Orchestrate eval: run sessions → collect → score
│   │   └── report.py         # Generate markdown/JSON result tables
│   └── baseline/
│       ├── __init__.py
│       ├── no_rag_agents.py  # Non-RAG baseline agent variants
│       ├── no_kgda.py        # Ablation: random concept selection
│       └── no_raa_ground.py  # Ablation: LLM-only scoring
├── tests/
│   ├── conftest.py               # Shared fixtures (mock KB, mock LLM, test ChromaDB)
│   ├── unit/
│   │   ├── test_kb_schema.py
│   │   ├── test_kb_retriever.py
│   │   ├── test_dma.py
│   │   ├── test_kgda.py
│   │   ├── test_qga.py
│   │   ├── test_raa.py
│   │   ├── test_fsa.py
│   │   └── test_state.py
│   ├── integration/
│   │   ├── test_graph_flow.py    # End-to-end graph with mock LLM
│   │   └── test_kb_chroma.py     # ChromaDB round-trip tests
│   └── evaluation/
│       └── test_metrics.py
├── scripts/
│   ├── seed_kb.py                # CLI: load concepts.json → ChromaDB
│   ├── run_interview.py          # CLI: interactive interview session
│   ├── run_evaluation.py         # CLI: full evaluation pipeline
│   └── run_baseline.py           # CLI: baseline + ablation runs
└── notebooks/
    ├── 01_kb_exploration.ipynb
    ├── 02_agent_prototyping.ipynb
    └── 03_results_analysis.ipynb
```

---

## 2. Phase-by-Phase Implementation (12-Week Timeline)

### Phase 1: Foundation — KB + Project Scaffold (Weeks 1–2)

**Deliverables:** Project scaffolding, KB with 30–40 DSA concepts, ChromaDB index, retrieval spot-checked.

#### Week 1

1. **Initialize project** with `pyproject.toml` (see Section 10 for dependencies).
2. **Implement `src/arag/kb/schema.py`** — ConceptEntry Pydantic model:

```python
class ConceptEntry(BaseModel):
    concept_id: str                          # e.g., "dsa-binary-search-tree"
    concept_name: str                        # e.g., "Binary Search Tree"
    definition: str                          # 2-4 sentence definition
    key_properties: list[str]                # Properties a correct answer must cover
    common_misconceptions: list[str]         # Documented error patterns
    example_correct_response: str            # Reference answer at expected level
    difficulty_level: Literal["beginner", "intermediate", "advanced"]
    related_concepts: list[str]              # List of concept_ids
```

3. **Author `data/kb/concepts.json`** — 30–40 DSA concept entries across three tiers:
   - **Beginner (10–12):** Arrays, Linked Lists, Stacks, Queues, Hash Tables, Basic Sorting (Bubble, Selection, Insertion), Linear Search, Binary Search, Strings, Recursion
   - **Intermediate (12–15):** BST, AVL Trees, Heaps, Graphs (BFS/DFS), Merge Sort, Quick Sort, Dynamic Programming intro, Greedy Algorithms, Two Pointers, Sliding Window, Backtracking
   - **Advanced (8–10):** Red-Black Trees, B-Trees, Trie, Segment Trees, Topological Sort, Dijkstra, Advanced DP (knapsack variants), Amortized Analysis, Union-Find

4. **Implement `src/arag/kb/loader.py`:**
   - `load_concepts(path) -> list[ConceptEntry]` — load and validate JSON
   - `embed_and_upsert(concepts, collection, openai_client)` — generate embeddings (text = concept_name + definition + key_properties), upsert to ChromaDB with full metadata

5. **Implement `src/arag/config.py`** using pydantic-settings:
   - Loads from `.env` (API keys) + `configs/default.yaml` (model params, session params)
   - Key fields: `openai_api_key`, `anthropic_api_key`, `llm_model`, `embedding_model`, `chroma_persist_dir`, `max_topics_per_session`, `max_depth_per_topic`, `similarity_top_k`

#### Week 2

6. **Implement `src/arag/kb/retriever.py`** — `KBRetriever` class:
   - `get_by_concept_id(concept_id) -> ConceptEntry | None` — metadata filter lookup
   - `search_by_response(response_text, top_k) -> list[tuple[ConceptEntry, float]]` — dense vector search
   - `get_related_concepts(concept_id) -> list[ConceptEntry]` — follow `related_concepts` links
   - `get_by_difficulty(difficulty, exclude_ids) -> list[ConceptEntry]` — filtered retrieval

7. **Write `scripts/seed_kb.py`** — CLI to load concepts.json into ChromaDB.
8. **Write tests:** `test_kb_schema.py`, `test_kb_retriever.py`, `test_kb_chroma.py`
9. **Manual spot-check:** run 5–10 sample queries, verify top-k relevance.

---

### Phase 2: Core Agents — DMA + KGDA (Weeks 3–4)

**Deliverables:** DMA and KGDA implemented and unit-tested. LangGraph state schema defined. Shared data models established.

#### Week 3

10. **Define shared data models** in `src/arag/models.py`:

```python
class IntentType(str, Enum):
    SUBSTANTIVE_RESPONSE = "substantive_response"
    CLARIFICATION_QUESTION = "clarification_question"
    NON_ANSWER = "non_answer"
    REQUEST_HINT = "request_hint"

class DMAAction(str, Enum):
    DELEGATE_KGDA = "delegate_kgda"
    DELEGATE_QGA = "delegate_qga"
    RESPOND_DIRECTLY = "respond_directly"
    MOVE_ON = "move_on"
    TRIGGER_FSA = "trigger_fsa"

class GapItem(BaseModel):
    property_text: str
    status: Literal["addressed_correctly", "incomplete", "incorrect", "not_addressed"]
    evidence: str = ""

class GapReport(BaseModel):
    concept_id: str
    concept_name: str
    items: list[GapItem]
    priority_gap: str | None = None
    retrieval_sources: list[str] = []

class ElementScore(BaseModel):
    element: str
    score: float  # 0.0–1.0
    justification: str
    grounding_source: str

class AssessmentRecord(BaseModel):
    concept_id: str
    question_text: str
    response_text: str
    element_scores: list[ElementScore]
    overall_score: float
    grounding_sources: list[str]
    misconceptions_detected: list[str]

class TopicState(BaseModel):
    concept_id: str
    concept_name: str
    depth: int = 0
    questions_asked: list[str] = []
    gap_reports: list[GapReport] = []
    assessments: list[AssessmentRecord] = []
    status: Literal["active", "completed", "skipped"] = "active"

class TopicFeedback(BaseModel):
    concept_name: str
    score: float
    strengths: list[str]
    gaps: list[str]
    recommendations: list[str]
    cited_sources: list[str]

class FeedbackReport(BaseModel):
    overall_summary: str
    topic_summaries: list[TopicFeedback]
    strengths: list[str]
    areas_for_improvement: list[str]
    kb_references: list[str]
```

11. **Define LangGraph state** in `src/arag/graph/state.py`:

```python
class InterviewState(TypedDict):
    # Conversation
    messages: Annotated[list, add_messages]

    # DMA deterministic state
    current_topic: TopicState | None
    topic_queue: list[str]              # concept_ids remaining
    topics_covered: list[TopicState]
    session_turn_count: int
    max_turns: int

    # DMA LLM outputs
    classified_intent: IntentType | None
    dma_action: DMAAction | None
    dma_direct_response: str | None

    # Agent outputs
    current_gap_report: GapReport | None
    current_assessment: AssessmentRecord | None
    pending_question: str | None
    expected_elements: list[str]

    # Session-level
    all_assessments: list[AssessmentRecord]
    feedback_report: FeedbackReport | None
    session_complete: bool
```

12. **Implement DMA** in `src/arag/agents/dma.py`:
    - **Deterministic methods** (no LLM):
      - `initialize_session(state)` — select initial topics from KB by difficulty distribution, populate `topic_queue`
      - `update_topic_state(state)` — update depth counters, mark topics complete, advance queue
      - `should_end_session(state)` — check max_turns, empty queue, or all topics at max_depth
    - **LLM methods:**
      - `classify_intent(state) -> IntentType` — classify candidate's last message
      - `decide_action(state) -> DMAAction` — given intent + gap report + topic state, decide next action
      - `generate_direct_response(state) -> str` — generate clarification, hint, or acknowledgment

#### Week 4

13. **Implement KGDA** in `src/arag/agents/kgda.py`:
    - `analyze(candidate_response, concept_id, expected_elements) -> GapReport`
    - Steps: retrieve `key_properties` + `example_correct_response` → LLM compares response against properties → classify each: addressed_correctly / incomplete / incorrect / not_addressed → identify `priority_gap`

14. **Write prompt templates** in `src/arag/prompts/dma_prompts.py` and `kgda_prompts.py`:
    - DMA intent classification: system prompt constraining output to exactly one IntentType
    - DMA action decision: system prompt with topic depth, remaining topics, gap context
    - KGDA analysis: system prompt with retrieved KB content, structured output format

15. **Write unit tests** for DMA (deterministic + mocked LLM) and KGDA.

---

### Phase 3: Remaining Agents — QGA + RAA + FSA (Weeks 5–6)

**Deliverables:** All five agents implemented and individually tested.

#### Week 5

16. **Implement QGA** in `src/arag/agents/qga.py`:
    - `generate_opening_question(concept_id) -> tuple[str, list[str]]` — retrieve definition + key_properties → generate open-ended question → return (question, expected_elements)
    - `generate_followup_question(gap_report, qa_history) -> tuple[str, list[str]]` — from `priority_gap`, retrieve concept content → generate targeted follow-up → ensure no repeat of addressed content

17. **Implement RAA** in `src/arag/agents/raa.py`:
    - `assess(candidate_response, expected_elements, concept_id) -> AssessmentRecord`
    - Steps: retrieve `example_correct_response` + `common_misconceptions` → LLM scores each element (0.0–1.0) with justification → check misconception matches → each score includes `grounding_source`

#### Week 6

18. **Implement FSA** in `src/arag/agents/fsa.py`:
    - `synthesize(assessments, topics_covered) -> FeedbackReport`
    - Steps: aggregate assessments by topic → for each gap, retrieve cited KB content → LLM generates per-topic summary (strengths + gaps + recommendations) → compile overall summary → all statements must cite `concept_ids`

19. **Write prompt templates** for QGA, RAA, FSA.
20. **Write unit tests** for QGA, RAA, FSA.

---

### Phase 4: Integration + Baseline (Week 7)

**Deliverables:** Full LangGraph pipeline running end-to-end. Non-RAG baseline built.

21. **Implement graph nodes** in `src/arag/graph/nodes.py`:
    - `dma_classify_intent(state) -> dict` — updates `classified_intent`
    - `dma_decide_action(state) -> dict` — updates `dma_action`
    - `run_kgda(state) -> dict` — updates `current_gap_report`
    - `run_qga(state) -> dict` — updates `pending_question` + `expected_elements`
    - `run_raa(state) -> dict` — updates `current_assessment` + `all_assessments`
    - `run_fsa(state) -> dict` — updates `feedback_report` + `session_complete`
    - `dma_respond_directly(state) -> dict` — updates `dma_direct_response` + `messages`
    - `dma_move_on(state) -> dict` — updates `current_topic`, `topic_queue`, `topics_covered`
    - `dma_deliver_question(state) -> dict` — packages `pending_question` into `messages`

22. **Build the graph** in `src/arag/graph/builder.py`:

```
Graph flow:
  [candidate_input]
    → classify_intent
    → decide_action
    → CONDITIONAL:
        delegate_kgda → run_kgda → run_raa → decide_action (loop)
        delegate_qga  → run_qga → deliver_question → END (wait for input)
        respond_directly → END (wait for input)
        move_on → run_qga → deliver_question → END (wait for input)
        trigger_fsa → run_fsa → END (session complete)
```

Key design decisions:
- **Hub-and-spoke pattern** with DMA as the hub
- **Interrupt pattern:** graph returns `END` when awaiting candidate input; session loop re-invokes with new message; LangGraph checkpointing preserves state
- **Cycle handling:** `run_kgda → run_raa → decide_action` loop allows DMA to inspect gap report before deciding follow-up vs. move-on; `max_depth_per_topic` prevents infinite loops
- **Routing function:** `route_by_action(state)` reads `state["dma_action"]` and returns edge key

23. **Implement non-RAG baseline** in `src/arag/baseline/no_rag_agents.py`:
    - `NoRAGQuestionGenerator` — fixed question bank per topic, no retrieval
    - `NoRAGAssessor` — LLM-only scoring without KB context
    - `NoRAGFeedback` — LLM-only feedback without KB citations
    - Build parallel graph variant via factory function in `builder.py`

24. **Write integration tests** — `test_graph_flow.py`: run 3–4 turns with mock LLM, verify state transitions.
25. **Write `scripts/run_interview.py`** for interactive CLI testing.

---

### Phase 5: Simulation + Evaluation (Weeks 8–9)

**Deliverables:** L1/L3 persona sessions run, raw results collected.

26. **Implement persona profiles** in `src/arag/simulation/personas.py`:

```python
class PersonaProfile(BaseModel):
    level: Literal["L1", "L3"]
    mastered_concepts: list[str]       # concept_ids fully known
    partial_concepts: list[str]        # incomplete knowledge
    unknown_concepts: list[str]        # cannot answer
    misconception_map: dict[str, list[str]]  # concept_id -> misconceptions to exhibit
```

- `build_l1_persona(concepts)` — masters ~60% of beginner, unknown on intermediate/advanced
- `build_l3_persona(concepts)` — full mastery beginner + intermediate, deliberate gaps in 3 advanced concepts

27. **Implement candidate simulator** in `src/arag/simulation/candidate.py`:
    - `SyntheticCandidate(profile, anthropic_client)` — uses Claude 3.5 Sonnet
    - `respond(interviewer_message) -> str` — system prompt encodes which concepts are known/partial/unknown, response style by level
    - **Circularity mitigation:** Claude for simulation, GPT-4o for agents (model family separation)

28. **Implement RAGAS adapter** in `src/arag/evaluation/ragas_adapter.py`:
    - `format_for_ragas(session_data) -> Dataset` — maps interview data to RAGAS format (question, answer, contexts, ground_truth)
    - `compute_ragas_metrics(dataset) -> dict[str, float]` — runs RAGAS `evaluate()` with faithfulness, answer_relevancy, context_relevancy

29. **Implement custom metrics** in `src/arag/evaluation/metrics.py`:

| Metric | RQ | Target | Description |
|--------|-----|--------|-------------|
| `gap_coverage_rate` | RQ1 | ≥ 0.80 | Proportion of planted gaps detected and probed |
| `redundancy_rate` | RQ1 | ≤ 0.10 | Questions overlapping already-addressed content |
| `element_precision_recall` | RQ2 | — | Element-level P/R vs. ground truth persona |
| `hallucination_rate` | RQ2 | — | Assessments citing incorrect/fabricated content |
| `evidence_citation_rate` | RQ3 | — | Feedback statements citing a specific KB entry |

30. **Implement evaluation runner** in `src/arag/evaluation/runner.py`:
    - Matrix: 4 system variants × 2 persona levels × 3 repetitions = 24 sessions
    - Variants: `full_system`, `baseline_no_rag`, `ablation_no_kgda`, `ablation_no_raa_ground`
    - Personas: `L1`, `L3`

---

### Phase 6: Ablations + Analysis (Week 10)

**Deliverables:** Ablation runs complete, results analyzed, failure modes documented.

31. **Implement ablation variants:**
    - `src/arag/baseline/no_kgda.py` — `RandomConceptSelector`: selects next concept randomly instead of gap-driven (isolates RQ1)
    - `src/arag/baseline/no_raa_ground.py` — `UngroundedAssessor`: LLM scores without retrieved ground truth (isolates RQ2)

32. **Build ablation graph variants** via factory in `builder.py`:
    ```python
    def build_graph(variant: Literal["full", "no_rag", "no_kgda", "no_raa_ground"]) -> CompiledGraph
    ```

33. **Run all experiments** via `scripts/run_evaluation.py`.
34. **Generate result tables** with `src/arag/evaluation/report.py`.

---

### Phase 7: Manuscript (Weeks 11–12)

Outside scope of code implementation. Results from Phase 6 feed into paper Sections 4–5.

---

## 3. Agent Design Details

### DMA — Dialogue Manager Agent

| Aspect | Detail |
|--------|--------|
| **Type** | Hybrid: deterministic state + LLM reasoning |
| **Input** | Candidate message + full session state |
| **Output** | Intent classification, action decision, direct responses |
| **LLM calls** | Intent classification, action decision, clarification/hint generation |
| **No LLM** | Session init, topic queue management, depth tracking, end-session check |
| **Constraint** | Sole interface with candidate — all other agents invisible |

### KGDA — Knowledge Gap Detection Agent

| Aspect | Detail |
|--------|--------|
| **Input** | Candidate response + concept_id + optional expected_elements |
| **Retrieves** | `key_properties` + `example_correct_response` for current concept |
| **Output** | `GapReport` with per-property status + priority_gap |
| **Key property** | Trigger = response quality, not candidate query |

### QGA — Question Generation Agent

| Aspect | Detail |
|--------|--------|
| **Input** | GapReport + Q&A history (or concept_id for opening question) |
| **Retrieves** | `definition`, `key_properties`, `common_misconceptions` |
| **Output** | Question text + expected_elements (forwarded to RAA) |
| **Constraint** | Must not repeat already-addressed content |

### RAA — Response Assessment Agent

| Aspect | Detail |
|--------|--------|
| **Input** | Candidate response + expected_elements (from QGA) + concept_id |
| **Retrieves** | `example_correct_response` + `common_misconceptions` |
| **Output** | `AssessmentRecord` with element-level scores + grounding_source |
| **Key property** | Correctness judged against retrieved reference, not LLM memory |

### FSA — Feedback Synthesis Agent

| Aspect | Detail |
|--------|--------|
| **Input** | All AssessmentRecords + topic coverage map |
| **Retrieves** | Cited KB content for each identified gap |
| **Output** | `FeedbackReport` with per-topic summary + KB references |
| **Trigger** | Once at session end |

---

## 4. Configuration

### `configs/default.yaml`

```yaml
llm:
  model: "gpt-4o"
  temperature: 0.3
  max_tokens: 2048

embedding:
  model: "text-embedding-3-small"

chroma:
  persist_dir: "./chroma_db"
  collection_name: "dsa_concepts"

session:
  max_topics: 5
  max_depth_per_topic: 3
  max_total_turns: 30

retrieval:
  top_k: 5
  similarity_threshold: 0.7
```

### `configs/evaluation.yaml`

```yaml
simulation:
  candidate_model: "claude-3-5-sonnet-20241022"
  repetitions: 3

personas:
  L1:
    beginner_mastery_ratio: 0.6
    intermediate_mastery_ratio: 0.0
    advanced_mastery_ratio: 0.0
  L3:
    beginner_mastery_ratio: 1.0
    intermediate_mastery_ratio: 1.0
    advanced_mastery_ratio: 0.7
    planted_gaps: 3

variants: [full, no_rag, no_kgda, no_raa_ground]

metrics:
  gap_coverage_target: 0.80
  redundancy_target: 0.10
```

Secrets (API keys) in `.env` only, never in YAML or committed files.

---

## 5. Testing Strategy

### Unit Tests (mocked LLM + mocked ChromaDB)

- Each agent tested in isolation with fixture responses
- DMA deterministic methods tested without mocks
- DMA LLM methods tested with mocked OpenAI client returning predefined JSON
- KB retriever tested against ephemeral ChromaDB collection

### Integration Tests

- `test_graph_flow.py`: compiled graph with mock LLM for 3–4 turns, verify state transitions
- `test_kb_chroma.py`: seed 5 concepts, verify embedding + retrieval round-trip

### Evaluation Tests

- `test_metrics.py`: unit test each custom metric with known inputs/outputs

### Shared Fixtures (`tests/conftest.py`)

- `mock_kb` — 5 sample DSA concepts
- `mock_chroma_collection` — ephemeral ChromaDB seeded with mock_kb
- `mock_openai_client` — returns predefined completions

---

## 6. Dependencies (`pyproject.toml`)

```toml
[project]
name = "arag-tech-interview"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain-openai>=0.2.0",
    "langchain-core>=0.3.0",
    "chromadb>=0.5.0",
    "openai>=1.40.0",
    "anthropic>=0.34.0",
    "ragas>=0.2.0",
    "pydantic>=2.8.0",
    "pydantic-settings>=2.4.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0",
    "ruff>=0.6.0",
    "mypy>=1.11",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## 7. Deliverables Summary

| Phase | Weeks | Key Deliverables |
|-------|-------|-----------------|
| 1 | 1–2 | Project scaffold, KB (30–40 concepts), ChromaDB index, retriever, config |
| 2 | 3–4 | DMA (hybrid), KGDA, shared models, LangGraph state, unit tests |
| 3 | 5–6 | QGA, RAA, FSA, all prompt templates, agent unit tests |
| 4 | 7 | LangGraph graph builder, full pipeline, non-RAG baseline, integration tests |
| 5 | 8–9 | Persona builder, Claude simulator, RAGAS adapter, custom metrics, eval runner |
| 6 | 10 | Ablation variants, full experiment runs, result tables, failure analysis |
| 7 | 11–12 | Manuscript completion (outside code scope) |
