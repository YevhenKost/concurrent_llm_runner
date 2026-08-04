# llm-batch-runner

Concurrent, caching, retrying batch runner for OpenAI-compatible chat
completion endpoints, with optional structured-output validation against
a Pydantic model.

## Features

- Fires requests concurrently (configurable `max_workers`) through the
  official `openai` SDK, so it works with OpenAI itself or any
  OpenAI-compatible `base_url` (vLLM, Together, Groq, local servers, ...).
- If you pass a Pydantic `base_model`, requests are made with a
  JSON-schema `response_format` and every response is validated against
  the model. Invalid output is retried, up to `n_retries` times.
- If no `base_model` is passed, it just does a normal completion and
  returns the raw text.
- Every sample is cached to disk as its own JSON file, keyed by a SHA-256
  hash of its input `messages`. On a re-run, already-cached samples are
  skipped unless `overwrite_existing=True`.
- Failures are logged (via the standard `logging` module) and returned
  in the result list rather than raised, so one bad sample never kills
  the whole batch.
- `LLMBatchRunner.load_cache(cache_dir)` is a static method that loads
  every cached record back into a dict, keyed by hash.

## Install

```bash
pip install llm-batch-runner
```

(Or, until published: `pip install -e .` from this directory.)

## Usage

```python
from pydantic import BaseModel
from llm_batch_runner import LLMBatchRunner

class Answer(BaseModel):
    reasoning: str
    value: int

conversations = [
    [{"role": "user", "content": "What is 12 * 7?"}],
    [{"role": "user", "content": "What is 9 * 9?"}],
    # ... as many as you like
]

runner = LLMBatchRunner(
    base_model=Answer,          # or None for plain text completions
    messages=conversations,     # List[List[dict]] — one conversation per sample
    max_workers=16,
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4o-mini",
    cache_dir="./llm_cache",
    n_retries=3,
    overwrite_existing=False,
)

results = runner.run()
print(results[0])
# OUTPUT:
```

Output:

```json
{
  "attempts": 1,
  "conversation": [
    {
      "role": "user",
      "content": "What is 12 * 7?"
    }
  ],
  "error": "None",
  "from_cache": "False",
  "hash": "f699170cc22a5e871fccdf3a414f35606d100c943d18387e58d75c0a1006b6e6",
  "index": 0,
  "output": {
    "reasoning": "12 * 7 can be calculated as (10 * 7) + (2 * 7) = 70 + 14 = 84.",
    "value": 84
  },
  "raw_output": "{\"reasoning\": \"12 * 7 can be calculated as (10 * 7) + (2 * 7) = 70 + 14 = 84.\", \"value\": 84}",
  "success": "True"
}
```

### Loading cached results later

```python
from llm_batch_runner import LLMBatchRunner

cache = LLMBatchRunner.load_cache("./llm_cache")
# List of dicts - loaded data from the cache
# [
# {"success": True, "output": {...}, "error": None, ...}, 
# ...
# ]
```

## Notes

- `messages` is a list of conversations — each conversation is itself the
  standard OpenAI `messages` list (`[{"role": ..., "content": ...}, ...]`).
- The cache key is a hash of the conversation only (not the model name),
  so if you change `model_name` and want fresh results, pass a different
  `cache_dir` or use `overwrite_existing=True`.
- Failed samples (that exhausted all retries) are also cached.
