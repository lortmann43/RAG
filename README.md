*This project has been created as part of the 42 curriculum by lortmann.*

# RAG against the machine

## Description

This project implements a local **Retrieval-Augmented Generation (RAG)** system that answers questions about the provided **vLLM 0.10.1** source code and documentation. Instead of relying solely on a language model's pretrained knowledge, it indexes the repository, retrieves evidence relevant to a question, and supplies that evidence to **Qwen/Qwen3-0.6B** to generate a grounded answer.

The system supports individual questions and JSON datasets, preserves exact source locations, and evaluates retrieval quality with **recall@k**. It uses Python, LangChain text splitters, BM25S, Hugging Face Transformers, Pydantic, Python Fire, and tqdm.

## Instructions

### Prerequisites

- Python **3.12 or newer** (the subject permits Python 3.10+, but this project's `pyproject.toml` requires 3.12+).
- [uv](https://docs.astral.sh/uv/) for installation and execution.
- The supplied vLLM 0.10.1 corpus under `data/raw/vllm-0.10.1/`.
- Sufficient disk space and memory for PyTorch and the Qwen3-0.6B model.

### Installation

From the repository root:

```bash
uv sync
```

Or use the Makefile:

```bash
make install
```

The Generator currently uses `local_files_only=True`. **Qwen3-0.6B must be downloaded to the Hugging Face cache before running generation offline.** On a machine with internet access, after `uv sync`:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B')"
```

Once the model is cached, it can be loaded without contacting Hugging Face.

### Expected directories

```text
data/
├── raw/vllm-0.10.1/                       # Input corpus
├── processed/                             # Saved BM25 index and chunk metadata
├── datasets/
│   ├── UnansweredQuestions/
│   └── AnsweredQuestions/
└── output/
    ├── search_results/UnansweredQuestions/
    └── search_results_and_answer/UnansweredQuestions/
```

The output directories are created as needed. The `--save_directory` option should be scoped to `UnansweredQuestions` or `AnsweredQuestions`, because the public datasets in those directories have identical filenames.

### Commands

The CLI is implemented using Python Fire. Run commands from the repository root with `uv run python -m src <command>`.

**Index the repository** (default chunk size: 2,000 characters):

```bash
uv run python -m src index --max_chunk_size 2000
```

**Search one question:**

```bash
uv run python -m src search "What endpoint loads a LoRA adapter dynamically?" --k 5
```

**Search an unanswered dataset:**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 \
  --save_directory data/output/search_results/UnansweredQuestions
```

**Evaluate retrieval against the corresponding ground truth:**

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

**Retrieve sources and answer a single question:**

```bash
uv run python -m src answer "What endpoint loads a LoRA adapter dynamically?" --k 3
```

**Generate answers from saved retrieval results:**

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --context_k 3
```

`answer_dataset` saves a checkpoint after each question and resumes previously completed questions by default. Supply `--resume=False` to regenerate the results. When changing `context_k`, regenerate the answers rather than reusing a checkpoint made with a different setting.

### Tests and code checks

```bash
uv run python -m pytest -v
make lint
```

The test suite covers chunk sizes and source offsets, IoU-based matching and recall, BM25 index persistence and retrieval, and checkpoint/resume behavior. The last local test run passed **16 tests**.

## System architecture

The pipeline follows the four RAG stages described in the subject:

```text
vLLM .py / .md / .txt files
    │
    ▼
Chunker ──► Indexer ──► Saved BM25 index + chunks metadata
                                      │
Question ──► Retriever ◄─────────────┘
                 │
                 ▼
          Ranked MinimalSource records
          (file path + character range)
                 │
                 ▼
           Context Builder
          (load original text)
                 │
                 ▼
        Qwen3-0.6B Generator
                 │
                 ▼
         Structured Pydantic answer
```

- `src/chunkers.py`: selects a splitter by file type and retains the original character offset.
- `src/indexer.py`: discovers supported files, builds the BM25 index, and saves/loads it alongside the chunk metadata.
- `src/retriever.py`: scores a question using the saved index and returns the top-ranked source locations.
- `src/context_builder.py`: reads exact substrings from the original source files.
- `src/generator.py`: builds a grounded chat prompt, checks the input token budget, and runs local Qwen3-0.6B inference.
- `src/pipeline.py`: coordinates retrieval, context building, and generation.
- `src/evaluator.py`: computes local recall@k against reference sources.
- `src/__main__.py`: exposes the six required CLI commands.
- `src/models/`: defines the Pydantic input and output schemas.

## Chunking strategy

Chunking is **file-type aware**:

- **Python (`.py`)**: LangChain `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` uses separators suited to Python source structure.
- **Markdown (`.md`)**: `MarkdownTextSplitter` uses Markdown-oriented separators.
- **Plain text (`.txt`)**: `RecursiveCharacterTextSplitter` uses generic text separators.

All splitters use `add_start_index=True` and `strip_whitespace=False` so that the extracted chunk can be matched against its source text. Files are opened with `newline=""` to preserve character positions. The saved metadata includes the source path and zero-based start index; the end index is the start plus chunk length (exclusive).

The default and maximum chunk size are **2,000 characters**. Overlap is set to `min(200, max_chunk_size // 10)`. This provides some continuity across chunk boundaries while respecting the per-source limit imposed by the evaluator.

The recorded public recall results below were obtained with the project's existing index configuration. A controlled comparison of different chunk sizes has **not** yet been recorded; no improvement from smaller chunks is claimed.

## Retrieval method

Retrieval uses **BM25**, a lexical document-ranking method implemented with **BM25S**:

1. At indexing time, chunk texts are tokenized with `bm25s.tokenize()`.
2. BM25S indexes the tokenized chunk collection and persists the index.
3. A question is tokenized using the same tokenizer.
4. BM25S calculates relevance scores and returns the highest-ranking chunk IDs.
5. Chunk IDs are mapped back to the original `file_path`, `first_character_index`, and `last_character_index`.

BM25 prioritizes chunks with matching query terms, taking term frequency and corpus-wide term rarity into account while accounting for document length. This is especially useful for literal code identifiers, paths, flags, and API names. Paraphrased questions can be harder for lexical matching.

Results with nonpositive BM25 scores are discarded. Retrieval results are serialized using the provided Pydantic models, including `StudentSearchResults` for datasets.

## Performance analysis

### Retrieval accuracy

The public documentation and code datasets were evaluated at **k = 5**, and the reported local recall values matched the supplied moulinette.

| Metric | Documentation | Code |
| --- | ---: | ---: |
| Recall@1 | 58.00% | 33.33% |
| Recall@3 | 77.00% | 47.47% |
| Recall@5 | **84.00%** | **55.56%** |
| Subject's Recall@5 requirement | 80% | 50% |

A reference source is counted as retrieved when its file path matches exactly and its character range overlaps a retrieved range with **Intersection over Union (IoU) >= 0.05**. Mean recall is calculated over questions with ground-truth sources.

### Runtime and generation quality

The subject imposes the following performance limits:

- Index the complete corpus within **5 minutes**.
- Retrieve answers for **200 questions within 90 seconds**.

These two end-to-end timing thresholds still need to be documented with a reproducible benchmark on the final evaluation machine. The recall results alone do not establish that the timing requirements pass.

Generation was tested locally on CPU. Qwen3-0.6B successfully answered questions such as the dynamic LoRA loading endpoint, although extra irrelevant context sometimes reduced answer accuracy. This is why the generation context count (`context_k`) is configurable independently of the retrieval count (`k`). An initial isolated Qwen test took approximately **20–30 seconds including model loading and generation**; this is not a representative full-dataset throughput measurement.

## Design decisions

- **BM25 rather than embedding-only retrieval:** provides lightweight, reproducible lexical retrieval without a separate vector service and meets the public recall thresholds.
- **Separate indexing, retrieval, context building, and generation:** each stage can be debugged and tested independently.
- **Original source offsets rather than rewritten text:** allows source locations to be validated by the official evaluator.
- **Pydantic models:** validate questions, sources, search results, and generated answers at pipeline boundaries.
- **Local Qwen3-0.6B:** follows the mandatory model requirement and avoids a hosted inference API once the model is cached.
- **Deterministic generation (`do_sample=False`):** prioritizes repeatability over creative phrasing.
- **Configurable `context_k`:** allows the system to retrieve more chunks than it sends to the smaller language model.
- **Incremental, atomic checkpoint writes:** prevent losing all finished answers if a batch is interrupted; resume skips completed question IDs.

## Challenges faced and solutions

**Preserving exact source offsets:** Text splitters can change whitespace and line endings, invalidating character ranges. The implementation preserves whitespace and newlines, stores each original start index, and verifies source-to-chunk equality in tests.

**Questions phrased differently from source code:** BM25 performs best on literal technical terms. A code-specific public dataset was used to measure whether retrieval remained adequate on the supplied questions. It reached 55.56% recall@5, above the 50% requirement, but semantic mismatches remain a limitation.

**Extra context sometimes harms generation:** For the LoRA endpoint example, three retrieved chunks produced the correct endpoint, while five chunks led Qwen to answer with an unrelated server command. A configurable `context_k` allows experiments with focused evidence without changing the stored top-k retrieval results.

**Reproducing exact commands:** Qwen sometimes abbreviated an `lm_eval` command as `...` even though the full command was in the first retrieved chunk. Restricting generation to the strongest evidence produced the complete command for this example.

**Intermittent Hugging Face connectivity:** Model and tokenizer loading were configured with `local_files_only=True` to avoid network attempts after the weights have been cached.

**Long-running batch generation:** Answers are saved after each completed question using a temporary file and replacement. Resume validates existing question IDs and source records before reusing results.

## Example usage

Example question:

```text
What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?
```

Example structured result (abbreviated to one of the retrieved sources):

```json
{
  "search_results": [
    {
      "question_id": "example-id",
      "question": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
          "first_character_index": 3463,
          "last_character_index": 5203
        }
      ],
      "answer": "The HTTP endpoint is /v1/load_lora_adapter."
    }
  ],
  "k": 1
}
```

This is an illustrative shortened output. Actual output contains the requested number of positively scored retrieval results, up to `k`.

## Resources

### Documentation and references

- [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — foundational description of retrieval-augmented generation.
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019) — background on probabilistic lexical ranking.
- [BM25S](https://github.com/xhluca/bm25s) — BM25 indexing and search library used in this project.
- [LangChain text splitters API](https://python.langchain.com/api_reference/text_splitters/index.html) — Python, Markdown, and recursive text splitting.
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B) — model information and usage.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — tokenizer, model loading, and inference.
- [Pydantic documentation](https://docs.pydantic.dev/latest/) — structured data validation.
- [Python Fire](https://github.com/google/python-fire) — command-line interface.
- [uv documentation](https://docs.astral.sh/uv/) — environment and dependency management.
- [pytest documentation](https://docs.pytest.org/en/stable/) — automated testing.

### Use of AI tools

ChatGPT was used during development as an **interactive learning and implementation aid**, including:

- Discussing the RAG architecture, BM25 ranking, chunking strategies, and recall@k.
- Suggesting and explaining implementations of pipeline components, Pydantic integration, CLI methods, and checkpoint/resume behavior.
- Helping troubleshoot Python typing, pytest imports, local model-loading warnings, and output-format issues.
- Proposing test cases, debugging experiments, and README structure.

Suggested code and design choices were reviewed, adapted, and tested locally. Functional behavior was checked with pytest, Flake8, mypy, manual answer-generation tests, and the provided moulinette. AI-generated suggestions were not treated as authoritative substitutes for the subject or test results.
