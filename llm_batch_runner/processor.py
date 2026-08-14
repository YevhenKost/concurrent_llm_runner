"""
Core implementation of LLMBatchRunner: a concurrent, caching, retrying
wrapper around the OpenAI-compatible chat completions API, with optional
structured-output validation against a Pydantic model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Tuple

from openai import OpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("llm_batch_runner")
if not logger.handlers:
    # Give the library a sane default handler so errors are visible
    # out of the box, without clobbering a user's own logging config.
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

Message = Dict[str, Any]
Conversation = List[Message]


@dataclass
class SampleResult:
    """Result of processing a single conversation."""

    index: int
    hash: str
    success: bool
    output: Optional[Union[dict, str]] = None
    raw_output: str = None
    error: Optional[str] = None
    from_cache: bool = False
    attempts: int = 0
    conversation: Conversation = None
    meta: List[Any] = None

    def to_dict(self) -> dict:
        return asdict(self)


def hash_messages(messages: Conversation) -> str:
    """Deterministically hash a conversation (list of message dicts)."""
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMBatchRunner:
    """
    Send a batch of chat-completion requests concurrently to an
    OpenAI-compatible endpoint, optionally enforcing structured output
    against a Pydantic model, with retries, per-sample disk caching
    (keyed by a hash of the input messages), and error reporting.

    Example
    -------
    >>> from pydantic import BaseModel
    >>> class Answer(BaseModel):
    ...     value: int
    >>> runner = LLMBatchRunner(
    ...     base_model=Answer,
    ...     messages=[[{"role": "user", "content": "2+2=?"}]],
    ...     max_workers=8,
    ...     api_key="sk-...",
    ...     base_url="https://api.openai.com/v1",
    ...     model_name="gpt-4o-mini",
    ...     cache_dir="./cache",
    ...     n_retries=3,
    ...     conversations_meta=None
    ... )
    >>> results = runner.run()
    """

    def __init__(
        self,
        base_model: Optional[Type[BaseModel]],
        messages: List[Conversation],
        max_workers: int,
        api_key: str,
        base_url: str,
        model_name: str,
        cache_dir: Union[str, Path],
        n_retries: int = 3,
        overwrite_existing: bool = False,
        temperature: Optional[float] = None,
        extra_create_kwargs: Optional[Dict[str, Any]] = None,
        show_progress: bool = True,
        conversations_meta: List[Any]|None = None,
        additional_validation_func = None,
        postprocessing_func=None
    ) -> None:
        if n_retries < 1:
            raise ValueError("n_retries must be >= 1")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        self.base_model = base_model
        self.messages = messages
        self.max_workers = max_workers
        self.model_name = model_name
        self.n_retries = n_retries
        self.overwrite_existing = overwrite_existing
        self.temperature = temperature
        self.extra_create_kwargs = extra_create_kwargs or {}
        self.show_progress = show_progress

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        self._progress_lock = threading.Lock()
        self._completed = 0

        self.additional_validation_func = additional_validation_func
        self.postprocessing_func = postprocessing_func

        self.conversations_meta = [None for _ in range(len(self.messages))]
        if conversations_meta is not None:
            assert isinstance(conversations_meta, list)
            assert len(conversations_meta) == len(messages)
            self.conversations_meta = conversations_meta


    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self) -> List[dict]:
        """
        Process all conversations concurrently and return a list of
        SampleResult objects, ordered the same way as the input.
        """
        total = len(self.messages)
        results: List[Optional[SampleResult]] = [None] * total

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_one, idx, conv): idx
                for idx, conv in enumerate(self.messages)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - safety net
                    logger.error(
                        "Unhandled exception while processing sample %d: %s",
                        idx,
                        exc,
                    )
                    result = SampleResult(
                        index=idx,
                        hash=hash_messages(self.messages[idx]),
                        success=False,
                        error=str(exc),
                        conversation=self.messages[idx],
                        meta=self.conversations_meta[idx]
                    )
                results[idx] = result
                self._report_progress(total)

        results = [
            x.to_dict() for x in results if x is not None
        ]
        return results

    @staticmethod
    def load_cache(cache_dir: Union[str, Path]) -> List[Dict]:
        """
        Load every cached JSON result from `cache_dir`.

        Returns a dict mapping {messages_hash: cached_record}, where
        cached_record has keys like "output", "success", "model_name", etc.
        """
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            return []


        cache = []
        for path in cache_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cache.append(json.load(f))
            except Exception as exc:
                logger.warning("Skipping unreadable cache file %s: %s", path, exc)
        return cache

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cached(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Cache file %s is corrupt, will reprocess: %s", path, exc)
            return None

    def _save_cache(self, key: str, record: dict) -> None:
        path = self._cache_path(key)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def _process_one(self, index: int, conversation: Conversation) -> SampleResult:
        key = hash_messages(conversation)

        if not self.overwrite_existing:
            cached = self._load_cached(key)
            if cached is not None:
                if cached.get("success", False):
                    return SampleResult(
                        index=index,
                        hash=key,
                        success=cached.get("success", True),
                        output=cached.get("output"),
                        error=cached.get("error"),
                        from_cache=True,
                        attempts=0,
                        conversation=conversation,
                        raw_output=cached.get("raw_output"),
                        meta=cached.get("meta")
                    )

        last_error: Optional[str] = None
        for attempt in range(1, self.n_retries + 1):
            try:
                output, raw_output = self._call_and_validate(conversation)
                if self.postprocessing_func is not None:
                    output = self.postprocessing_func(output, conversation)
                record = {
                    "success": True,
                    "output": output,
                    "error": None,
                    "model_name": self.model_name,
                    "attempts": attempt,
                    "conversation": conversation,
                    "raw_output": raw_output,
                    "meta": self.conversations_meta[index]
                }
                self._save_cache(key, record)
                return SampleResult(
                    index=index,
                    hash=key,
                    success=True,
                    output=output,
                    attempts=attempt,
                    conversation=conversation,
                    raw_output=raw_output
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Sample %d (hash=%s) attempt %d/%d failed: %s",
                    index,
                    key[:8],
                    attempt,
                    self.n_retries,
                    last_error,
                )

        logger.error(
            "Sample %d (hash=%s) failed after %d attempts: %s",
            index,
            key[:8],
            self.n_retries,
            last_error,
        )
        # # Cache the failure too, so a re-run without overwrite_existing
        # doesn't silently skip a sample that never succeeded.
        # record = {
        #     "success": False,
        #     "output": None,
        #     "error": last_error,
        #     "model_name": self.model_name,
        #     "attempts": self.n_retries,
        #     "conversation": conversation,
        #     "raw_output": None,
        #     "meta": self.conversations_meta[index]
        # }
        # self._save_cache(key, record)
        return SampleResult(
            index=index,
            hash=key,
            success=False,
            error=last_error,
            attempts=self.n_retries,
            raw_output=None,
            conversation=conversation,
            output=None
        )

    def _call_and_validate(self, conversation: Conversation) -> Tuple[Union[dict, str], str]:

        # Structured output
        if self.base_model is not None:
            create_kwargs: Dict[str, Any] = dict(
                model=self.model_name,
                messages=conversation,
                response_format=self.base_model,
                **self.extra_create_kwargs,
            )
            if self.temperature is not None:
                create_kwargs["temperature"] = self.temperature

            response = self.client.beta.chat.completions.parse(**create_kwargs)
            message = response.choices[0].message

            if hasattr(message, "refusal"):
                if message.refusal:
                    raise ValueError(f"Model refused to answer: {message.refusal}")

            if message.parsed is None:
                raise ValueError("Model did not return a parsed structured output")

            json_model_output = message.parsed.model_dump()
            validated_json_model_output = self._validate(json_model_output, conversation)
            return validated_json_model_output, message.content

        # Regular (non-structured) completion
        create_kwargs = dict(
            model=self.model_name,
            messages=conversation,
            **self.extra_create_kwargs,
        )
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**create_kwargs)
        return response.choices[0].message.content, response.choices[0].message.content

    def _validate(self, extracted_json, conversation) -> dict:

        try:
            instance = self.base_model.model_validate_json(json.dumps(extracted_json))
        except Exception as exc:
            raise ValueError(f"Structured output validation failed: {exc}") from exc

        if self.additional_validation_func is not None:
            try:
                is_valid = self.additional_validation_func(extracted_json, conversation)
                if not is_valid:
                    raise ValueError(f"Structured output validation failed via CUSTOM FUNCTION: {extracted_json}")
            except Exception as exc:
                raise ValueError(f"Structured output validation failed: {exc}") from exc


        return instance.model_dump()

    def _report_progress(self, total: int) -> None:
        if not self.show_progress:
            return
        with self._progress_lock:
            self._completed += 1
            print(f"\rProcessed {self._completed}/{total}", end="", flush=True)
            if self._completed == total:
                print()
