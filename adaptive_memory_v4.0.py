"""
Open WebUI Adaptive Memory Filter v4.0

This filter provides intelligent memory management for Open WebUI conversations,
including memory extraction, deduplication, vector-based retrieval, and background
summarization tasks.
"""

__version__ = "4.0.1"
__all__ = ["Filter"]

# Standard library imports
import asyncio
import difflib
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Union,
)
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Third-party imports
import aiohttp
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# Regex pattern for extracting memory content between tags and memory bank
_MEMORY_CONTENT_PATTERN = re.compile(
    r"\[Tags:[^\]]*\]\s*(.*?)\s*\[Memory Bank:[^\]]*\]",
    re.DOTALL,
)

# Embedding model imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore

# Metrics & Monitoring Imports
try:
    from prometheus_client import Counter, Histogram  # type: ignore
except ImportError:
    # Fallback: define dummy Counter/Histogram if prometheus_client not installed
    class _NoOpMetric:
        def __init__(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

    Counter = Histogram = _NoOpMetric  # type: ignore

# OpenWebUI Imports
try:
    from open_webui.config import DATA_DIR
except ImportError:
    from pathlib import Path

    DATA_DIR = Path("/app/backend/data")

try:
    from open_webui.models.memories import Memories
    from open_webui.models.users import Users
except ImportError:
    Memories = None  # type: ignore[assignment]
    Users = None     # type: ignore[assignment]

try:
    from open_webui.routers.memories import add_memory, AddMemoryForm
except ImportError:
    add_memory = None
    AddMemoryForm = None  # type: ignore

# Vector Database Client for Synchronization
try:
    from open_webui.retrieval.vector.factory import VECTOR_DB_CLIENT
except ImportError:
    VECTOR_DB_CLIENT = None

# Define Prometheus metrics
EMBEDDING_REQUESTS = Counter(
    "adaptive_memory_embedding_requests_total",
    "Total number of embedding requests",
    ["provider"],
)
EMBEDDING_ERRORS = Counter(
    "adaptive_memory_embedding_errors_total",
    "Total number of embedding errors",
    ["provider"],
)
EMBEDDING_LATENCY = Histogram(
    "adaptive_memory_embedding_latency_seconds",
    "Latency of embedding generation",
    ["provider"],
)

RETRIEVAL_REQUESTS = Counter(
    "adaptive_memory_retrieval_requests_total",
    "Total number of get_relevant_memories calls",
    [],
)
RETRIEVAL_ERRORS = Counter(
    "adaptive_memory_retrieval_errors_total", "Total number of retrieval errors", []
)
RETRIEVAL_LATENCY = Histogram(
    "adaptive_memory_retrieval_latency_seconds",
    "Latency of get_relevant_memories execution",
    [],
)

# Constants
DEFAULT_LLM_TIMEOUT = 120  # seconds
DEFAULT_BACKGROUND_ERROR_SLEEP = 60  # seconds
NOTIFICATION_QUEUE_MAXLEN = 500


# --- Advanced Mock Infrastructure for Router Compatibility ---
class MockConfig:
    """Mock configuration for OpenWebUI router compatibility."""

    def __init__(self):
        # Flags required by router.add_memory
        self.ENABLE_MEMORIES = True
        self.USER_PERMISSIONS = {"features": {"memories": True}}


class MockAppState:
    """Mock application state for OpenWebUI router compatibility."""

    def __init__(self, embedding_function):
        self.config = MockConfig()
        self.EMBEDDING_FUNCTION = embedding_function


class MockApp:
    """Mock application for OpenWebUI router compatibility."""

    def __init__(self, embedding_function):
        self.state = MockAppState(embedding_function)


class MockState:
    """Mock state for OpenWebUI router compatibility."""

    def __init__(self, user):
        self.user = user


class MockRequest:
    """Mock request for OpenWebUI router dependency injection."""

    def __init__(self, user, embedding_function):
        self.app = MockApp(embedding_function)
        self.state = MockState(user)
        self.user = user


# Set up logging with versioned adapter
_raw_logger = logging.getLogger("openwebui.plugins.adaptive_memory")
if not _raw_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    _raw_logger.addHandler(handler)
    _raw_logger.setLevel(logging.INFO)


class AMAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return "[AM v%s] %s" % (__version__, msg), kwargs


logger = AMAdapter(_raw_logger, {})


# ------------------------------------------------------------------------------
# Data Models and Helper Classes
# ------------------------------------------------------------------------------


class MemoryOperation(BaseModel):
    """Model for memory operations."""

    operation: Literal["NEW", "UPDATE", "DELETE"]
    id: Optional[str] = None
    content: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    memory_bank: Optional[str] = None
    confidence: Optional[float] = None


class LocalAddMemoryForm(BaseModel):
    content: str


class ErrorManager:
    """Centralized error tracking and reporting with thread safety."""

    def __init__(self):
        self.counters: Dict[str, int] = {
            "embedding_errors": 0,
            "llm_call_errors": 0,
            "json_parse_errors": 0,
            "memory_crud_errors": 0,
        }
        self._lock_obj = None

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazy initialization of async lock to avoid wrong event loop binding.

        NOTE: This is thread-safe under asyncio's cooperative scheduling model.
        Python's async/await only yields at await points, so the check-then-set
        pattern cannot race within a single event loop. Multi-thread access to
        the same asyncio.Lock() is not supported by this design.
        """
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    async def increment(self, counter_name: str) -> None:
        """Increment error counter in a thread-safe manner."""
        async with self._lock:
            self.counters[counter_name] = self.counters.get(counter_name, 0) + 1

    def get_counters(self) -> Dict[str, int]:
        """Get current error counters."""
        return self.counters.copy()


class JSONParser:
    """Robust JSON parsing utilities."""

    @staticmethod
    def extract_and_parse(text: str) -> Union[List[Any], Dict[str, Any], None]:
        """Extract and parse JSON from text, handling code blocks and raw content."""
        if not text:
            return None

        # 1. Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Extract from code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Extract from raw brackets (greedy for complete JSON structures)
        bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(1))
            except json.JSONDecodeError:
                pass

        return None


class LRUCache:
    """A simple LRU (Least Recently Used) cache with bounded size and TTL.

    Entries are evicted when the cache reaches max_size or when TTL expires.
    Most recently accessed items are kept, oldest items are removed first.
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 0):
        """Initialize LRU cache with maximum size and optional TTL.

        Args:
            max_size: Maximum number of entries to keep in cache
            ttl_seconds: Time-to-live in seconds (0 = no TTL)
        """
        self._cache = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock_obj = None

    @property
    def _lock(self) -> asyncio.Lock:
        """Lazy initialization of async lock to avoid wrong event loop binding.

        NOTE: This is thread-safe under asyncio's cooperative scheduling model.
        Python's async/await only yields at await points, so the check-then-set
        pattern cannot race within a single event loop.
        """
        if self._lock_obj is None:
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    async def get(self, key: str) -> Optional[np.ndarray]:
        """Get value from cache, moving it to end (most recently used).

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found and not expired, None otherwise
        """
        async with self._lock:
            if key in self._cache:
                value, ts = self._cache[key]
                if self._ttl <= 0 or (time.time() - ts) <= self._ttl:
                    self._cache.move_to_end(key)
                    return value
                else:
                    del self._cache[key]
            return None

    async def set(self, key: str, value: np.ndarray) -> None:
        """Set value in cache, evicting oldest entry if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def put(self, key: str, value: np.ndarray) -> None:
        """Alias for set() - for compatibility with existing code.

        Args:
            key: Cache key
            value: Value to cache
        """
        await self.set(key, value)


# ------------------------------------------------------------------------------
# Embedding Management
# ------------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def get_embedding(
        self, text: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[np.ndarray]:
        """Get embedding for a single text."""
        pass

    @abstractmethod
    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        """Get embeddings for a batch of texts."""
        pass


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using SentenceTransformer."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                logger.info("Loading local embedding model: %s", model_name)
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.exception("Failed to load local SentenceTransformer model: %s", e)

    async def get_embedding(
        self, text: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[np.ndarray]:
        """Get embedding for a single text using local model."""
        if not self.model:
            return None
        try:
            # Run blocking call in executor
            loop = asyncio.get_running_loop()
            model = self.model  # Extract to local variable for type narrowing
            embedding = await loop.run_in_executor(
                None, lambda: model.encode(text, normalize_embeddings=True)
            )
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.exception("Local embedding error: %s", e)
            return None

    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        """Get embeddings for a batch of texts using local model."""
        if not self.model or not texts:
            return [None] * len(texts)
        try:
            loop = asyncio.get_running_loop()
            model = self.model  # Extract to local variable for type narrowing
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(
                    texts, normalize_embeddings=True, show_progress_bar=False
                ),
            )
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.exception("Local batch embedding error: %s", e)
            return [None] * len(texts)


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible API embedding provider."""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 30,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def _do_embedding_request(
        self, text: str, session: aiohttp.ClientSession, timeout: aiohttp.ClientTimeout
    ) -> Optional[np.ndarray]:
        """Internal method to perform the actual embedding HTTP request."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % self.api_key,
        }
        data = {"input": text, "model": self.model_name}
        async with session.post(
            self.api_url, json=data, headers=headers, timeout=timeout
        ) as response:
            if response.status == 200:
                res_json = await response.json()
                if "data" in res_json and len(res_json["data"]) > 0:
                    emb = res_json["data"][0]["embedding"]
                    return np.array(emb, dtype=np.float32)
            return None

    async def _do_embeddings_batch_request(
        self, texts: List[str], session: aiohttp.ClientSession, timeout: aiohttp.ClientTimeout
    ) -> List[Optional[np.ndarray]]:
        """Internal method to perform the actual batch embedding HTTP request."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer %s" % self.api_key,
        }
        data = {"input": texts, "model": self.model_name}
        async with session.post(
            self.api_url, json=data, headers=headers, timeout=timeout
        ) as response:
            if response.status == 200:
                res_json = await response.json()
                if "data" in res_json:
                    # Correct indexing: map by the 'index' field
                    results = [None] * len(texts)
                    for item in res_json["data"]:
                        idx = item.get("index")
                        embedding_data = item.get("embedding")
                        if (
                            idx is not None
                            and 0 <= idx < len(results)
                            and embedding_data is not None
                        ):
                            results[idx] = np.array(embedding_data, dtype=np.float32)
                        elif idx is not None and 0 <= idx < len(results):
                            logger.warning(
                                "Missing embedding data for item at index %d in "
                                "batch response",
                                idx,
                            )
                    return results
            return [None] * len(texts)

    async def get_embedding(
        self, text: str, session: Optional[aiohttp.ClientSession] = None
    ) -> Optional[np.ndarray]:
        """Get embedding for a single text via API with retry logic."""
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        for attempt in range(self.max_retries + 1):
            try:
                if session:
                    return await self._do_embedding_request(text, session, timeout)
                async with aiohttp.ClientSession() as new_session:
                    return await self._do_embedding_request(text, new_session, timeout)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Embedding API attempt %d/%d failed, retrying in %ss: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        self.retry_delay,
                        e,
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.exception("API embedding error after retries: %s", e)
                    return None
            except Exception as e:
                logger.exception("Unexpected API embedding error: %s", e)
                return None
        return None

    async def get_embeddings_batch(
        self, texts: List[str], session: Optional[aiohttp.ClientSession] = None
    ) -> List[Optional[np.ndarray]]:
        """Get embeddings for a batch of texts via API with retry logic."""
        timeout = aiohttp.ClientTimeout(total=self.timeout * 2)  # Longer for batch
        for attempt in range(self.max_retries + 1):
            try:
                if session:
                    return await self._do_embeddings_batch_request(texts, session, timeout)
                async with aiohttp.ClientSession() as new_session:
                    return await self._do_embeddings_batch_request(texts, new_session, timeout)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Batch embedding API attempt %d/%d failed, retrying in %ss: %s",
                        attempt + 1,
                        self.max_retries + 1,
                        self.retry_delay,
                        e,
                    )
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.exception("API batch embedding error after retries: %s", e)
                    return [None] * len(texts)
            except Exception as e:
                logger.exception("Unexpected API batch embedding error: %s", e)
                return [None] * len(texts)
        return [None] * len(texts)


class EmbeddingManager:
    """Manages embedding generation, caching, and persistence."""

    def __init__(self, get_valves: Callable[[], Any], error_manager: ErrorManager):
        self.get_valves = get_valves
        self.error_manager = error_manager
        self.cache = LRUCache()  # Bounded LRU cache (default max_size=10000)
        self.provider: Optional[EmbeddingProvider] = None
        self._current_provider_type = None
        self._session: Optional[aiohttp.ClientSession] = None
        # Use regular dict instead of WeakValueDictionary to avoid premature GC
        self._locks: Dict[str, asyncio.Lock] = {}
        self._init_lock_obj = None

    @property
    def _init_lock(self) -> asyncio.Lock:
        """Lazy initialization of async lock to avoid wrong event loop binding.

        NOTE: This is thread-safe under asyncio's cooperative scheduling model.
        Python's async/await only yields at await points, so the check-then-set
        pattern cannot race within a single event loop.
        """
        if self._init_lock_obj is None:
            self._init_lock_obj = asyncio.Lock()
        return self._init_lock_obj

    def _get_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create a lock for the given user_id (idempotent).

        Locks are never deleted to avoid race conditions where a lock
        could be deleted while another coroutine is waiting to acquire it.
        The memory overhead is minimal (one lock per active user).
        """
        return self._locks.setdefault(user_id, asyncio.Lock())

    async def cleanup(self):
        """Clean up resources like the shared HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _ensure_session(self):
        """Ensure a shared aiohttp session exists (async-safe)."""
        async with self._init_lock:
            if not self._session or self._session.closed:
                self._session = aiohttp.ClientSession()

    async def _ensure_provider(self):
        """Ensure embedding provider is initialized (async-safe)."""
        async with self._init_lock:
            valves = self.get_valves()
            # Initialize if not set or if provider type changed
            if (
                not self.provider
                or self._current_provider_type != valves.embedding_provider_type
            ):
                self._current_provider_type = valves.embedding_provider_type
                if valves.embedding_provider_type == "local":
                    self.provider = LocalEmbeddingProvider(valves.embedding_model_name)
                elif valves.embedding_provider_type == "openai_compatible":
                    self.provider = OpenAICompatibleEmbeddingProvider(
                        valves.embedding_api_url,
                        valves.embedding_api_key,
                        valves.embedding_model_name,
                        timeout=valves.embedding_timeout,
                        max_retries=valves.max_retries,
                        retry_delay=valves.retry_delay,
                    )

    async def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text with metrics tracking."""
        if not text:
            return None

        EMBEDDING_REQUESTS.labels(self.get_valves().embedding_provider_type).inc()
        start = time.perf_counter()

        if not self.provider:
            await self._ensure_provider()

        if not self.provider:
            return None

        await self._ensure_session()
        emb = await self.provider.get_embedding(text, session=self._session)

        if emb is not None:
            EMBEDDING_LATENCY.labels(
                self.get_valves().embedding_provider_type
            ).observe(time.perf_counter() - start)
        else:
            await self.error_manager.increment("embedding_errors")
            EMBEDDING_ERRORS.labels(self.get_valves().embedding_provider_type).inc()

        return emb

    async def get_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[np.ndarray]]:
        """Get embeddings for a batch of texts."""
        if not self.provider:
            await self._ensure_provider()

        if not self.provider:
            return [None] * len(texts)

        await self._ensure_session()
        return await self.provider.get_embeddings_batch(texts, session=self._session)

    async def store_embedding_persistent(
        self,
        user_id: str,
        memory_id: str,
        memory_text: str,
        embedding: np.ndarray,
    ) -> None:
        """Store memory embedding in a persistent JSON file for reload across restarts."""
        async with self._get_lock(user_id):
            try:
                # Use data directory for persistence
                cache_dir = os.path.join(DATA_DIR, "cache", "embeddings")
                await asyncio.to_thread(os.makedirs, cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "%s_embeddings.json" % user_id)

                # Load existing cache
                cache = {}
                if await asyncio.to_thread(os.path.exists, cache_file):
                    try:

                        def _load():
                            with open(cache_file, "r") as f:
                                return json.load(f)

                        cache = await asyncio.to_thread(_load)
                    except Exception as e:
                        logger.warning(
                            "Error loading embedding cache, starting fresh: %s", e
                        )

                # Convert numpy array to list for JSON storage
                embedding_list = (
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                )

                # Store embedding with metadata (Ensure ID is string for JSON key)
                cache[str(memory_id)] = {
                    "embedding": embedding_list,
                    "model": self.get_valves().embedding_model_name,
                    "provider": self.get_valves().embedding_provider_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                # Save cache atomically
                def _save_atomic(data, cache_file):
                    fd, tmp_file = tempfile.mkstemp(
                        dir=os.path.dirname(os.path.abspath(cache_file))
                    )
                    try:
                        with os.fdopen(fd, "w") as f:
                            json.dump(data, f)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_file, cache_file)
                    except Exception:
                        try:
                            os.unlink(tmp_file)
                        except OSError:
                            pass
                        raise

                await asyncio.to_thread(_save_atomic, cache, cache_file)
                logger.debug(
                    "Stored embedding for memory %s in persistent cache", memory_id
                )
            except Exception as e:
                logger.warning(
                    "Failed to store embedding in persistent cache for memory %s: %s",
                    memory_id,
                    e,
                )

    async def store_embeddings_batch_persistent(
        self, user_id: str, ids: List[str], texts: List[str], embeddings: List[np.ndarray]
    ) -> None:
        """Store multiple embeddings in a single persistent JSON file operation."""
        if not ids:
            return

        async with self._get_lock(user_id):
            try:
                cache_dir = os.path.join(DATA_DIR, "cache", "embeddings")
                await asyncio.to_thread(os.makedirs, cache_dir, exist_ok=True)
                cache_file = os.path.join(cache_dir, "%s_embeddings.json" % user_id)

                # Load existing cache
                cache = {}
                if await asyncio.to_thread(os.path.exists, cache_file):
                    try:

                        def _load():
                            with open(cache_file, "r") as f:
                                return json.load(f)

                        cache = await asyncio.to_thread(_load)
                    except Exception as e:
                        logger.warning(
                            "Error loading embedding cache for batch store: %s", e
                        )

                # Update cache with new embeddings
                for memory_id, embedding in zip(ids, embeddings):
                    if embedding is None:
                        continue

                    embedding_list = (
                        embedding.tolist()
                        if isinstance(embedding, np.ndarray)
                        else embedding
                    )
                    cache[str(memory_id)] = {
                        "embedding": embedding_list,
                        "model": self.get_valves().embedding_model_name,
                        "provider": self.get_valves().embedding_provider_type,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                # Save cache atomically
                def _save_atomic(data, cache_file):
                    fd, tmp_file = tempfile.mkstemp(
                        dir=os.path.dirname(os.path.abspath(cache_file))
                    )
                    try:
                        with os.fdopen(fd, "w") as f:
                            json.dump(data, f)
                            f.flush()
                            os.fsync(f.fileno())
                        os.replace(tmp_file, cache_file)
                    except Exception:
                        try:
                            os.unlink(tmp_file)
                        except OSError:
                            pass
                        raise

                await asyncio.to_thread(_save_atomic, cache, cache_file)
                logger.info(
                    "Batched stored %d embeddings in persistent cache for user %s",
                    len(ids),
                    user_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to store batch embeddings in persistent cache: %s", e
                )

    async def load_embedding_persistent(
        self, user_id: str, memory_id: str
    ) -> Optional[np.ndarray]:
        """Load a stored embedding from the persistent JSON file."""
        result = None
        async with self._get_lock(user_id):
            try:
                cache_dir = os.path.join(DATA_DIR, "cache", "embeddings")
                cache_file = os.path.join(cache_dir, "%s_embeddings.json" % user_id)

                if not await asyncio.to_thread(os.path.exists, cache_file):
                    result = None
                else:
                    # Load cache
                    def _load():
                        with open(cache_file, "r") as f:
                            return json.load(f)

                    cache = await asyncio.to_thread(_load)

                    # Ensure ID is handled as string for JSON key lookup
                    memory_id_str = str(memory_id)
                    if memory_id_str in cache:
                        embedding_data = cache[memory_id_str]
                        embedding_list = embedding_data["embedding"]
                        embedding = np.array(embedding_list, dtype=np.float32)

                        # Validate model compatibility
                        valves = self.get_valves()
                        stored_model = embedding_data.get("model")
                        stored_provider = embedding_data.get("provider")

                        if (
                            stored_model != valves.embedding_model_name
                            or stored_provider != valves.embedding_provider_type
                        ):
                            logger.debug(
                                "Cache miss for %s: Model/provider changed",
                                memory_id_str,
                            )
                            result = None
                        else:
                            result = embedding
                    else:
                        result = None
            except Exception as e:
                logger.warning(
                    "Error loading embedding from persistent cache for memory %s: %s",
                    memory_id,
                    e,
                )
                result = None
        return result

    async def get_embedding_with_persistence(
        self, text: str, user_id: str, memory_id: str
    ) -> Optional[np.ndarray]:
        """Get embedding with full caching hierarchy: memory -> persistent -> generate."""
        if not text:
            return None

        # Ensure memory_id is a string for consistent caching
        memory_id_str = str(memory_id)

        # 1. Check in-memory cache first
        cached_emb = await self.cache.get(memory_id_str)
        if cached_emb is not None:
            return cached_emb

        # 2. Check persistent cache
        persistent_emb = await self.load_embedding_persistent(user_id, memory_id_str)
        if persistent_emb is not None:
            # Cache in memory for this session
            await self.cache.set(memory_id_str, persistent_emb)
            return persistent_emb

        # 3. Generate new embedding
        new_emb = await self.get_embedding(text)
        if new_emb is not None:
            # Cache in memory
            await self.cache.set(memory_id_str, new_emb)
            # Store persistently
            await self.store_embedding_persistent(
                user_id, memory_id_str, text, new_emb
            )

        return new_emb


# ------------------------------------------------------------------------------
# Memory Pipeline
# ------------------------------------------------------------------------------


class MemoryPipeline:
    """Core logic for extracting, retrieving, and processing memories."""

    def __init__(
        self,
        get_valves: Callable[[], Any],
        embedding_manager: EmbeddingManager,
        error_manager: ErrorManager,
    ):
        self.get_valves = get_valves
        self.embedding_manager = embedding_manager
        self.error_manager = error_manager

    @property
    def valves(self):
        """Get current valves - always returns fresh values."""
        return self.get_valves()

    async def identify_memories(
        self,
        user_message: str,
        context_memories: Optional[List[Dict[str, Any]]] = None,
        query_llm_func: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Identify potential memories from user message using LLM.

        Args:
            user_message: The user's message text
            context_memories: Optional context memories for the identification
            query_llm_func: Function to query the LLM

        Returns:
            List of valid memory operation dictionaries
        """
        if not user_message:
            return []

        # Construct prompt
        system_prompt = self.valves.memory_identification_prompt
        now = datetime.now(timezone.utc)
        system_prompt += "\n\nCurrent Date: %s" % now.strftime("%Y-%m-%d %H:%M:%S")

        user_prompt = "User Message: %s" % user_message
        if context_memories:
            user_prompt += "\n\nContext Memories:\n" + "\n".join(
                ["- %s" % m.get("content", "") for m in context_memories]
            )

        # Call LLM
        if not query_llm_func:
            return []

        try:
            response = await query_llm_func(system_prompt, user_prompt)
            if not response:
                return []

            # Parse JSON
            data = JSONParser.extract_and_parse(response)
            if not isinstance(data, list):
                return []

            # Validate and filter
            valid_ops = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                op = item.get("operation")
                content = item.get("content")
                confidence = item.get("confidence", 0.0)

                if op in ["NEW", "UPDATE"] and content:
                    if confidence >= self.valves.min_confidence_threshold:
                        valid_ops.append(item)
                elif op == "DELETE" and item.get("id"):
                    # DELETE doesn't require content
                    valid_ops.append(item)

            return valid_ops

        except Exception as e:
            await self.error_manager.increment("llm_call_errors")
            logger.exception("Identify memories failed: %s", e)
            return []

    async def get_relevant_memories(
        self, query: str, user_id: str, all_memories: List[Any]
    ) -> List[Any]:
        """Retrieve relevant memories using vector similarity.

        Args:
            query: The search query
            user_id: User ID to search memories for
            all_memories: List of all user memories

        Returns:
            List of relevant memories sorted by similarity
        """
        if not query or not all_memories:
            return []

        RETRIEVAL_REQUESTS.inc()
        start_time = time.perf_counter()

        # 1. Vector Search
        try:
            query_embedding = await self.embedding_manager.get_embedding(query)
            if query_embedding is None:
                return []
        except Exception as e:
            RETRIEVAL_ERRORS.inc()
            logger.exception("Error generating query embedding: %s", e)
            return []

        scored_memories = []

        # Batch embedding for memories without cached embeddings
        mem_objects = []
        texts_to_embed = []
        ids_to_embed = []

        for mem in all_memories:
            # Handle object vs dict
            mem_content = mem.content if hasattr(mem, "content") else mem.get("content")
            mem_id = mem.id if hasattr(mem, "id") else mem.get("id")

            if not mem_id or not mem_content:
                continue

            # Check in-memory cache first
            cached_emb = await self.embedding_manager.cache.get(mem_id)
            if cached_emb is not None:
                sim = self._cosine_similarity(query_embedding, cached_emb)
                if sim >= self.valves.vector_similarity_threshold:
                    scored_memories.append((sim, mem))
            else:
                # Check persistent cache
                persistent_emb = await self.embedding_manager.load_embedding_persistent(
                    user_id, mem_id
                )
                if persistent_emb is not None:
                    # Cache in memory for this session
                    await self.embedding_manager.cache.set(mem_id, persistent_emb)
                    sim = self._cosine_similarity(query_embedding, persistent_emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem))
                else:
                    # Need to generate embedding
                    mem_objects.append(mem)
                    texts_to_embed.append(mem_content)
                    ids_to_embed.append(mem_id)

        if texts_to_embed:
            logger.info(
                "Generating embeddings for %d memories (using cache for %d)",
                len(texts_to_embed),
                len(all_memories) - len(texts_to_embed),
            )
            # Batch generate
            new_embeddings = await self.embedding_manager.get_embeddings_batch(
                texts_to_embed
            )
            for i, emb in enumerate(new_embeddings):
                if emb is not None:
                    # Update in-memory cache
                    await self.embedding_manager.cache.set(ids_to_embed[i], emb)
                    # Score
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.valves.vector_similarity_threshold:
                        scored_memories.append((sim, mem_objects[i]))

            # Store all newly generated embeddings persistently in one go
            if any(e is not None for e in new_embeddings):
                await self.embedding_manager.store_embeddings_batch_persistent(
                    user_id, ids_to_embed, texts_to_embed, new_embeddings
                )
        else:
            logger.info("Using cached embeddings for all %d memories", len(all_memories))

        # Sort by similarity
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        top_memories = [
            mem for sim, mem in scored_memories[: self.valves.related_memories_n]
        ]

        RETRIEVAL_LATENCY.observe(time.perf_counter() - start_time)
        return top_memories

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity value between 0 and 1
        """
        if v1.shape != v2.shape:
            logger.debug(
                "Cosine similarity dimension mismatch: %s vs %s", v1.shape, v2.shape
            )
            return 0.0
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison.

        Removes punctuation, articles, intensifiers, and extra spaces.

        Args:
            text: Text to normalize

        Returns:
            Normalized text string
        """
        # Remove punctuation, extra spaces, convert to lowercase
        normalized = re.sub(r"[^\w\s]", "", text.strip().lower())

        # Remove articles (a, an, the)
        normalized = re.sub(r"\b(a|an|the)\b", "", normalized)

        # Remove intensifiers
        normalized = re.sub(
            r"\b(really|very|quite|pretty|so|totally|absolutely)\b",
            "",
            normalized,
        )

        # Clean up extra spaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _extract_raw_content(self, memory_content: str) -> str:
        """Extract raw content from formatted memory string.

        Handles memory format: [Tags: ...] CONTENT [Memory Bank: ...] [Confidence: ...]

        Args:
            memory_content: Formatted memory content string

        Returns:
            Extracted raw content or original string if parsing fails
        """
        if not memory_content:
            return ""

        # Use regex pattern to extract content between tags and memory bank
        m = _MEMORY_CONTENT_PATTERN.search(memory_content)
        return m.group(1).strip() if m else memory_content

    async def process_memory_operations(
        self,
        operations: List[Dict[str, Any]],
        user_id: str,
        skip_deduplication: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute valid memory operations (NEW, UPDATE, DELETE).

        Args:
            operations: List of memory operation dictionaries
            user_id: User ID to perform operations for
            skip_deduplication: Whether to skip deduplication checks

        Returns:
            List of successfully executed operations
        """
        # Fetch full user object for Router DI (MockRequest)
        user_obj = None
        if Users is not None:
            user_obj = await asyncio.to_thread(Users.get_user_by_id, user_id)

        # Define embedding function helper once (outside loop to avoid shadowing)
        async def _embedding_fn(text: str, user=None) -> Optional[np.ndarray]:
            return await self.embedding_manager.get_embedding(text)

        # Check capabilities once
        router_available = add_memory and (AddMemoryForm or LocalAddMemoryForm)

        success_ops = []
        for op in operations:
            try:
                kind = op.get("operation")
                content = op.get("content")

                if kind == "NEW" and content:
                    # Incorporate metadata into content string for storage
                    tags = op.get("tags", [])
                    bank = op.get("memory_bank", "General")

                    # Deduplication check (skip for summaries)
                    dedup_embedding = None
                    if self.valves.deduplicate_memories and not skip_deduplication:
                        is_dupe, dedup_embedding = await self._is_duplicate(
                            content, user_id
                        )
                        if is_dupe:
                            logger.info("Skipping duplicate memory (length: %d)", len(content))
                            continue

                    # Format: [Tags: tag1, tag2] Content [Memory Bank: Bank] [Confidence: X.XX]
                    tags_str = ", ".join(tags) if tags else "none"
                    confidence = op.get("confidence", 1.0)

                    final_content = "[Tags: %s] %s [Memory Bank: %s] [Confidence: %.2f]" % (
                        tags_str,
                        content,
                        bank,
                        confidence,
                    )

                    # Add memory - using Router add_memory for Vector Indexing
                    try:
                        mem_obj = None
                        try:
                            # Try the high-level router function
                            logger.info(
                                "Attempting to add memory via router add_memory "
                                "(Vector-Aware)..."
                            )

                            if router_available:
                                # Use imported AddMemoryForm if available, otherwise local
                                FormClass = AddMemoryForm if AddMemoryForm else LocalAddMemoryForm
                                form = FormClass(content=final_content)

                                # Prepare Mock Request for Dependency Injection
                                req = MockRequest(user_obj, _embedding_fn)

                                # Call add_memory with correct signature
                                mem_obj = await add_memory(
                                    request=req, form_data=form, user=user_obj
                                )
                            else:
                                raise ImportError(
                                    "Router add_memory not successfully imported"
                                )

                        except Exception as add_err:
                            logger.warning(
                                "Router add_memory failed (%s), falling back to "
                                "insert_new_memory",
                                add_err,
                            )
                            if Memories is not None:
                                mem_obj = await asyncio.to_thread(
                                    Memories.insert_new_memory, user_id, final_content
                                )
                            else:
                                logger.error("Memories model not available for fallback insert")

                        memory_id = getattr(mem_obj, "id", None)
                        if memory_id:
                            success_ops.append(op)
                            logger.info(
                                "Memory saved (ID: %s) [Bank: %s] [Confidence: %.2f]",
                                memory_id,
                                bank,
                                confidence,
                            )

                            # Cache the embedding from deduplication check
                            if dedup_embedding is not None:
                                logger.debug(
                                    "Caching embedding from deduplication check for "
                                    "memory %s",
                                    memory_id,
                                )
                                await self.embedding_manager.cache.set(
                                    str(memory_id), dedup_embedding
                                )
                                # Persist it immediately
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id, str(memory_id), content, dedup_embedding
                                )
                        else:
                            logger.error(
                                "Failed to save memory - no memory ID returned"
                            )
                            await self.error_manager.increment("memory_crud_errors")
                    except Exception as ins_err:
                        await self.error_manager.increment("memory_crud_errors")
                        logger.error("Failed to insert memory: %s", ins_err)

                elif kind == "UPDATE" and op.get("id") and op.get("content"):
                    try:
                        memory_id = op.get("id")
                        if not memory_id:
                            logger.warning("UPDATE operation missing memory ID")
                            continue
                        new_content = op["content"]

                        # Update in database
                        updated_memory = None
                        if Memories is not None:
                            updated_memory = await asyncio.to_thread(
                                Memories.update_memory_by_id_and_user_id,
                                memory_id,
                                user_id,
                                new_content,
                            )

                        if updated_memory:
                            # Regenerate embedding for updated content
                            new_embedding = await self.embedding_manager.get_embedding(
                                new_content
                            )

                            if new_embedding is not None:
                                # Update in-memory cache
                                await self.embedding_manager.cache.set(
                                    str(memory_id), new_embedding
                                )

                                # Update persistent cache
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id, str(memory_id), new_content, new_embedding
                                )

                                # Update vector database if available
                                if VECTOR_DB_CLIENT:
                                    try:
                                        await asyncio.to_thread(
                                            VECTOR_DB_CLIENT.upsert,
                                            collection_name="user-memory-%s" % user_id,
                                            items=[
                                                {
                                                    "id": str(memory_id),
                                                    "text": new_content,
                                                    "vector": (
                                                        new_embedding.tolist()
                                                        if hasattr(
                                                            new_embedding, "tolist"
                                                        )
                                                        else new_embedding
                                                    ),
                                                    "metadata": {
                                                        "updated_at": (
                                                            updated_memory.updated_at
                                                            if hasattr(
                                                                updated_memory,
                                                                "updated_at",
                                                            )
                                                            else None
                                                        ),
                                                    },
                                                }
                                            ],
                                        )
                                        logger.info(
                                            "Memory updated in vector DB (ID: %s)",
                                            memory_id,
                                        )
                                    except Exception as vec_err:
                                        logger.warning(
                                            "Failed to update memory in vector DB: %s",
                                            vec_err,
                                        )

                            success_ops.append(op)
                            logger.info("Memory updated (ID: %s)", memory_id)
                        else:
                            logger.warning(
                                "Memory not found for update (ID: %s)", memory_id
                            )

                    except Exception as upd_err:
                        logger.exception("Failed to update memory: %s", upd_err)

                elif kind == "DELETE" and op.get("id"):
                    try:
                        memory_id = op["id"]

                        # Delete from database (user-scoped)
                        result = False
                        if Memories is not None:
                            result = await asyncio.to_thread(
                                Memories.delete_memory_by_id_and_user_id, memory_id, user_id
                            )
                        if result:
                            # Delete from vector database if available
                            if VECTOR_DB_CLIENT:
                                try:
                                    await asyncio.to_thread(
                                        VECTOR_DB_CLIENT.delete,
                                        collection_name="user-memory-%s" % user_id,
                                        ids=[str(memory_id)],
                                    )
                                    logger.info(
                                        "Memory deleted from vector DB (ID: %s)",
                                        memory_id,
                                    )
                                except Exception as vec_err:
                                    logger.warning(
                                        "Failed to delete memory from vector DB: %s",
                                        vec_err,
                                    )

                            success_ops.append(op)
                            logger.info("Memory deleted (ID: %s)", memory_id)
                        else:
                            logger.warning(
                                "Failed to delete memory (ID: %s) - not found or "
                                "permission denied",
                                memory_id,
                            )
                    except Exception as del_err:
                        logger.exception("Failed to delete memory: %s", del_err)

            except Exception as e:
                await self.error_manager.increment("memory_crud_errors")
                logger.exception("Memory operation failed: %s", e)

        return success_ops

    async def _is_duplicate(
        self,
        text: str,
        user_id: str,
        exclude_id: Optional[str] = None,
        all_memories_override: Optional[List[Any]] = None,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Check if the given text is a duplicate of existing memories.

        Args:
            text: Text to check for duplicates
            user_id: User ID to check memories for
            exclude_id: Optional memory ID to exclude from comparison
            all_memories_override: Optional pre-fetched memories list

        Returns:
            Tuple of (is_duplicate: bool, embedding: Optional[np.ndarray])
            The embedding is returned so it can be cached after successful save.
        """
        if not text or not self.valves.deduplicate_memories:
            return False, None

        try:
            # Get all existing memories for the user (or use override)
            if all_memories_override is not None:
                all_memories = all_memories_override
            else:
                if Memories is not None:
                    all_memories = await asyncio.to_thread(
                        Memories.get_memories_by_user_id, user_id
                    )
                else:
                    all_memories = []

            if all_memories is None:
                logger.error("Failed to retrieve memories for deduplication check")
                return False, None
            if not all_memories:
                return False, None

            if self.valves.use_embeddings_for_deduplication:
                # Use embedding-based similarity
                new_embedding = await self.embedding_manager.get_embedding(text)
                if new_embedding is None:
                    logger.warning(
                        "Could not generate embedding for duplicate check, "
                        "falling back to text similarity"
                    )
                    # Fallback to text similarity - no embedding to cache
                    is_dup = await self._check_text_similarity(
                        text, all_memories, exclude_id=exclude_id
                    )
                    return is_dup, None

                # Check similarity against existing memories
                for memory in all_memories:
                    memory_id = (
                        memory.id if hasattr(memory, "id") else memory.get("id")
                    )

                    # SAFETY: Skip comparing memory against itself
                    if exclude_id and str(memory_id) == str(exclude_id):
                        continue

                    memory_content = (
                        memory.content
                        if hasattr(memory, "content")
                        else memory.get("content")
                    )

                    # Extract raw content from formatted memory for comparison
                    raw_memory_content = self._extract_raw_content(memory_content)

                    # Check for exact match first (ignoring punctuation and case)
                    if self._normalize_text(text) == self._normalize_text(
                        raw_memory_content
                    ):
                        logger.info("Exact match found for memory %s", memory_id)
                        return True, new_embedding

                    # Use raw content for embedding comparison
                    content_for_embedding = raw_memory_content
                    # Check in-memory cache first
                    existing_embedding = await self.embedding_manager.cache.get(
                        memory_id
                    )
                    if existing_embedding is None:
                        # Check persistent cache
                        existing_embedding = await (
                            self.embedding_manager.load_embedding_persistent(
                                user_id, memory_id
                            )
                        )
                        if existing_embedding is not None:
                            # Cache in memory for this session
                            await self.embedding_manager.cache.set(
                                memory_id, existing_embedding
                            )
                        else:
                            # Generate embedding for existing memory using raw content
                            existing_embedding = await (
                                self.embedding_manager.get_embedding(
                                    content_for_embedding
                                )
                            )
                            if existing_embedding is not None:
                                # Cache in memory and store persistently
                                await self.embedding_manager.cache.set(
                                    memory_id, existing_embedding
                                )
                                await self.embedding_manager.store_embedding_persistent(
                                    user_id,
                                    memory_id,
                                    content_for_embedding,
                                    existing_embedding,
                                )

                    if existing_embedding is not None:
                        # Calculate similarity
                        similarity = self._cosine_similarity(
                            new_embedding, existing_embedding
                        )

                        # Use embedding similarity threshold from valves
                        if similarity >= self.valves.embedding_similarity_threshold:
                            logger.info(
                                "Duplicate detected via embeddings (similarity: %.3f) "
                                "for memory %s",
                                similarity,
                                memory_id,
                            )
                            return True, new_embedding
                    else:
                        logger.warning(
                            "Could not generate embedding for existing memory %s",
                            memory_id,
                        )
            else:
                # Use text-based similarity - no embedding to cache
                is_dup = await self._check_text_similarity(
                    text, all_memories, exclude_id=exclude_id
                )
                return is_dup, None

            return (
                False,
                new_embedding if self.valves.use_embeddings_for_deduplication else None,
            )

        except Exception as e:
            logger.exception("Error during duplicate check: %s", e)
            # If deduplication fails, err on the side of caution and keep the memory
            return False, None

    async def _check_text_similarity(
        self, text: str, all_memories: List[Any], exclude_id: Optional[str] = None
    ) -> bool:
        """Check for text-based similarity using difflib.

        Args:
            text: Text to compare
            all_memories: List of memories to compare against
            exclude_id: Optional memory ID to exclude

        Returns:
            True if a duplicate is detected, False otherwise
        """
        normalized_text = self._normalize_text(text)

        for memory in all_memories:
            memory_id = memory.id if hasattr(memory, "id") else memory.get("id")

            # SAFETY: Skip comparing memory against itself
            if exclude_id and str(memory_id) == str(exclude_id):
                continue

            memory_content = (
                memory.content if hasattr(memory, "content") else memory.get("content")
            )

            # Extract raw content from formatted memory for comparison
            raw_memory_content = self._extract_raw_content(memory_content)

            # Calculate text similarity using normalized raw content
            normalized_raw = self._normalize_text(raw_memory_content)
            similarity = difflib.SequenceMatcher(
                None, normalized_text, normalized_raw
            ).ratio()

            if similarity >= self.valves.similarity_threshold:
                logger.info(
                    "Duplicate detected via text similarity (similarity: %.3f) "
                    "for memory %s",
                    similarity,
                    memory_id,
                )
                return True

        return False

    async def cluster_and_summarize(
        self, user_id: str, query_llm_func: Callable
    ) -> Optional[str]:
        """Find clusters of memories and summarize them.

        Args:
            user_id: User ID to process memories for
            query_llm_func: Function to query LLM for summarization

        Returns:
            Summary message string if successful, None otherwise
        """
        logger.info("Starting summarization for user %s", user_id)

        # 1. Fetch memories
        try:
            memories = []
            if Memories is not None:
                memories = await asyncio.to_thread(
                    Memories.get_memories_by_user_id, user_id
                )
            logger.info(
                "Found %d memories for user %s",
                len(memories) if memories else 0,
                user_id,
            )

            if not memories:
                logger.info(
                    "No memories found for user %s, skipping summarization", user_id
                )
                return None

            if len(memories) < self.valves.summarization_min_cluster_size:
                logger.info(
                    "Only %d memories found for user %s, need at least %d for clustering",
                    len(memories),
                    user_id,
                    self.valves.summarization_min_cluster_size,
                )
                return None
        except Exception as e:
            logger.exception("Summarization fetch failed: %s", e)
            return None

        # 2. Get embeddings (use cache when possible)
        logger.info("Processing embeddings for %d memories", len(memories))
        contents = [m.content for m in memories]
        ids = [m.id for m in memories]

        # Check cache hierarchy: memory -> persistent -> generate
        embeddings = []
        uncached_indices = []
        uncached_contents = []

        for i, (memory_id, content) in enumerate(zip(ids, contents, strict=True)):
            # Check in-memory cache first
            cached_embedding = await self.embedding_manager.cache.get(memory_id)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                # Check persistent cache
                persistent_embedding = await (
                    self.embedding_manager.load_embedding_persistent(user_id, memory_id)
                )
                if persistent_embedding is not None:
                    # Cache in memory for this session
                    await self.embedding_manager.cache.set(
                        memory_id, persistent_embedding
                    )
                    embeddings.append(persistent_embedding)
                else:
                    # Need to generate
                    embeddings.append(None)  # Placeholder
                    uncached_indices.append(i)
                    uncached_contents.append(content)

        # Generate embeddings for uncached memories
        if uncached_contents:
            logger.info(
                "Generating embeddings for %d uncached memories (using cache for %d)",
                len(uncached_contents),
                len(memories) - len(uncached_contents),
            )
            new_embeddings = await self.embedding_manager.get_embeddings_batch(
                uncached_contents
            )

            # Update cache and embeddings list
            for idx, new_emb in zip(uncached_indices, new_embeddings, strict=True):
                if new_emb is not None:
                    embeddings[idx] = new_emb
                    # Cache in memory
                    await self.embedding_manager.cache.set(ids[idx], new_emb)

            # Store persistently in batch
            await self.embedding_manager.store_embeddings_batch_persistent(
                user_id,
                [str(ids[idx]) for idx in uncached_indices],
                uncached_contents,
                new_embeddings,  # Already aligned with uncached_contents
            )
        else:
            logger.info("Using cached embeddings for all %d memories", len(memories))
            new_embeddings = []  # No new embeddings when all are cached

        valid_indices = [i for i, e in enumerate(embeddings) if e is not None]
        newly_generated_count = (
            len([e for e in new_embeddings if e is not None]) if new_embeddings else 0
        )
        logger.info(
            "Ready for clustering: %d valid embeddings (%d from cache, %d newly "
            "generated)",
            len(valid_indices),
            len(memories) - len(uncached_contents),
            newly_generated_count,
        )

        if len(valid_indices) < self.valves.summarization_min_cluster_size:
            logger.info(
                "Only %d valid embeddings, need at least %d for clustering",
                len(valid_indices),
                self.valves.summarization_min_cluster_size,
            )
            return None

        # 3. Simple Greedy Clustering
        logger.info(
            "Starting clustering with similarity threshold %s",
            self.valves.summarization_similarity_threshold,
        )
        clusters = []
        visited = set()

        for i in valid_indices:
            if i in visited:
                continue

            cluster = [i]
            visited.add(i)
            vec_i = embeddings[i]

            for j in valid_indices:
                if j in visited:
                    continue

                vec_j = embeddings[j]
                sim = self._cosine_similarity(vec_i, vec_j)

                if sim >= self.valves.summarization_similarity_threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) >= self.valves.summarization_min_cluster_size:
                clusters.append(cluster)
                logger.info(
                    "Found cluster with %d memories (similarity >= %s)",
                    len(cluster),
                    self.valves.summarization_similarity_threshold,
                )

        logger.info("Found %d clusters ready for summarization", len(clusters))

        # 4. Summarize Clusters
        total_summarized = 0
        for cluster_indices in clusters:
            try:
                # Enforce max cluster size
                cluster_memories = [
                    memories[i]
                    for i in cluster_indices[:self.valves.summarization_max_cluster_size]
                ]
                cluster_text = "\n".join(["- %s" % m.content for m in cluster_memories])

                summary = await query_llm_func(
                    self.valves.summarization_memory_prompt,
                    "Memories to summarize:\n%s" % cluster_text,
                )

                if summary:
                    # 5. Execute Changes Transactionally: Save before Delete
                    op = {
                        "operation": "NEW",
                        "content": summary,
                        "tags": ["summary"],
                        "memory_bank": "General",
                        "confidence": 1.0,
                    }

                    # Process the new summary first (skip deduplication)
                    success_ops = await self.process_memory_operations(
                        [op], user_id, skip_deduplication=True
                    )

                    if success_ops:
                        logger.info(
                            "Summarization: New consolidated summary saved successfully. "
                            "Now removing %d source memories.",
                            len(cluster_memories),
                        )
                        # ONLY delete old if saving the new summary succeeded
                        for m in cluster_memories:
                            try:
                                memory_id = str(m.id)

                                # Delete from database (user-scoped)
                                deleted = False
                                if Memories is not None:
                                    deleted = await asyncio.to_thread(
                                        Memories.delete_memory_by_id_and_user_id,
                                        memory_id,
                                        user_id,
                                    )

                                if deleted:
                                    # Delete from vector database if available
                                    if VECTOR_DB_CLIENT:
                                        try:
                                            await asyncio.to_thread(
                                                VECTOR_DB_CLIENT.delete,
                                                collection_name="user-memory-%s" % user_id,
                                                ids=[memory_id],
                                            )
                                            logger.debug(
                                                "Summarization: Deleted memory %s from vector DB",
                                                memory_id,
                                            )
                                        except Exception as vec_err:
                                            logger.warning(
                                                "Summarization: Failed to delete memory %s "
                                                "from vector DB: %s",
                                                memory_id,
                                                vec_err,
                                            )
                                else:
                                    logger.warning(
                                        "Summarization: Failed to delete source memory %s - "
                                        "not found or permission denied",
                                        memory_id,
                                    )

                            except Exception as del_err:
                                logger.error(
                                    "Summarization: Failed to delete source memory %s: %s",
                                    m.id,
                                    del_err,
                                )

                        logger.info(
                            "Summarized %d memories into new summary (Confidence 1.0)",
                            len(cluster_memories),
                        )
                        total_summarized += len(cluster_memories)
                    else:
                        logger.error(
                            "Summarization: Failed to save new summary. Aborting source "
                            "memory deletion to prevent data loss."
                        )

            except Exception as e:
                await self.error_manager.increment("memory_crud_errors")
                logger.exception("Memory operation failed: %s", e)

        return (
            "Consolidated %d memories into summaries." % total_summarized
            if total_summarized
            else None
        )


class TaskManager:
    """Manages background tasks."""

    def __init__(self, filter_instance: Any):
        self.filter = filter_instance
        self.tasks: Set[asyncio.Task] = set()

    def start_tasks(self) -> bool:
        """Attempt to start background tasks. Returns True if successful."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                "TaskManager: No running event loop found. Tasks will be retried on "
                "next request."
            )
            return False

        # Kill rogue ghost tasks from previous versions before starting new ones
        scavenger_task = asyncio.create_task(self._scavenge_rogue_tasks())
        self.tasks.add(scavenger_task)
        scavenger_task.add_done_callback(self.tasks.discard)

        valves = self.filter.valves
        logger.info(
            "Starting background tasks: summarization=%s",
            valves.enable_summarization_task,
        )

        if valves.enable_summarization_task:
            task = asyncio.create_task(self.filter._summarize_old_memories_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_error_logging_task:
            task = asyncio.create_task(self.filter._log_error_counters_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        if valves.enable_vector_cleanup_task:
            task = asyncio.create_task(self.filter._cleanup_vectors_loop())
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)

        logger.info("Background tasks started: %d active tasks", len(self.tasks))
        return True

    async def stop_tasks(self):
        """Stop all background tasks."""
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

    async def _scavenge_rogue_tasks(self):
        """Find and terminate any orphaned background tasks from previous versions."""
        logger.info("TaskManager: Starting rogue task scavenger...")
        current_task = asyncio.current_task()
        all_tasks = asyncio.all_tasks()

        scavenged_count = 0
        for task in all_tasks:
            if task == current_task:
                continue

            # Look for tasks running functions related to adaptive memory loops
            task_repr = repr(task)
            # We target the specific function names used in v3.1 and v4.0
            ghost_indicators = [
                "_summarize_old_memories_loop",
                "_deduplicate_memories_loop",
                "_remove_duplicate_memories",
                "function_adaptive_memory_v31",
            ]

            if any(indicator in task_repr for indicator in ghost_indicators):
                # If it's not one of OUR currently tracked tasks, it's a ghost
                if task not in self.tasks:
                    logger.warning(
                        "TaskManager: Found potentially rogue ghost task: %s. "
                        "Requesting cancellation.",
                        task_repr,
                    )
                    task.cancel()
                    scavenged_count += 1

        if scavenged_count > 0:
            logger.info(
                "TaskManager: Scavenger requested cancellation of %d rogue tasks.",
                scavenged_count,
            )
        else:
            logger.info("TaskManager: No rogue tasks detected.")


# ------------------------------------------------------------------------------
# Main Filter Class
# ------------------------------------------------------------------------------


class Filter:
    """Main Open WebUI Filter class for adaptive memory management."""

    # --------------------------------------------------------------------------
    # Configuration / Valves (PRESERVED EXACTLY)
    # --------------------------------------------------------------------------
    class Valves(BaseModel):
        """Configuration valves for the filter"""

        # Embedding Model Configuration
        embedding_provider_type: Literal["local", "openai_compatible"] = Field(
            default="local",
            description="Type of embedding provider ('local' for SentenceTransformer "
            "or 'openai_compatible' for API)",
        )
        embedding_model_name: str = Field(
            default="all-MiniLM-L6-v2",
            description="Name of the embedding model to use (e.g., 'all-MiniLM-L6-v2', "
            "'text-embedding-3-small')",
        )
        embedding_api_url: Optional[str] = Field(
            default=None,
            description="API endpoint URL for the embedding provider (required if type "
            "is 'openai_compatible')",
        )
        embedding_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the embedding provider (required if type is "
            "'openai_compatible')",
        )

        # Background Task Management Configuration
        enable_summarization_task: bool = Field(
            default=True,
            description="Enable or disable the background memory summarization task",
        )
        summarization_interval: int = Field(
            default=7200,
            description="Interval in seconds between memory summarization runs",
        )
        enable_error_logging_task: bool = Field(
            default=True,
            description="Enable or disable the background error counter logging task",
        )
        error_logging_interval: int = Field(
            default=1800,
            description="Interval in seconds between error counter log entries",
        )

        enable_vector_cleanup_task: bool = Field(
            default=True,
            description="Enable or disable the background vector cleanup task",
        )
        vector_cleanup_interval: int = Field(
            default=7200,
            description="Interval in seconds between vector cleanup runs (removes "
            "orphaned embeddings)",
        )

        # Summarization Configuration
        summarization_min_cluster_size: int = Field(
            default=3,
            description="Minimum number of memories in a cluster for summarization",
        )
        summarization_similarity_threshold: float = Field(
            default=0.7,
            description="Threshold for considering memories related when using "
            "embedding similarity",
        )
        summarization_max_cluster_size: int = Field(
            default=8,
            description="Maximum memories to include in one summarization batch",
        )
        summarization_memory_prompt: str = Field(
            default="""You are a memory summarization assistant. Your task is to combine related memories about a user into a concise, comprehensive summary.

Given a set of related memories about a user, create a single paragraph that:
1. Captures all key information from the individual memories
2. Resolves any contradictions (prefer newer information)
3. Maintains specific details when important
4. Removes redundancy
5. Presents the information in a clear, concise format

Focus on preserving the user's:
- Explicit preferences
- Identity details
- Goals and aspirations
- Relationships
- Possessions
- Behavioral patterns

Your summary should be factual, concise, and maintain the same tone as the original memories.
Produce a single paragraph summary of approximately 50-100 words that effectively condenses the information.

Example:
Individual memories:
- "User likes to drink coffee in the morning"
- "User prefers dark roast coffee"
- "User mentioned drinking 2-3 cups of coffee daily"

Good summary:
"User is a coffee enthusiast who drinks 2-3 cups daily, particularly enjoying dark roast varieties in the morning."

Analyze the following related memories and provide a concise summary.""",
            description="System prompt for summarizing clusters of related memories",
        )

        # Filtering & Saving Configuration
        enable_json_stripping: bool = Field(
            default=True,
            description="Attempt to strip non-JSON text before/after the main JSON "
            "object/array from LLM responses.",
        )
        enable_fallback_regex: bool = Field(
            default=True,
            description="If primary JSON parsing fails, attempt a simple regex "
            "fallback to extract at least one memory.",
        )
        enable_short_preference_shortcut: bool = Field(
            default=True,
            description="If JSON parsing fails for a short message containing "
            "preference keywords, directly save the message content.",
        )
        short_preference_no_dedupe_length: int = Field(
            default=100,
            description="If a NEW memory's content length is below this threshold "
            "and contains preference keywords, skip deduplication checks to avoid "
            "false positives.",
        )
        preference_keywords_no_dedupe: str = Field(
            default="favorite,love,like,prefer,enjoy",
            description="Comma-separated keywords indicating user preferences that, "
            "when present in a short statement, trigger deduplication bypass.",
        )
        blacklist_topics: Optional[str] = Field(
            default=None,
            description="Optional: Comma-separated list of topics to ignore during "
            "memory extraction",
        )
        filter_trivia: bool = Field(
            default=True,
            description="Enable filtering of trivia/general knowledge memories "
            "after extraction",
        )
        whitelist_keywords: Optional[str] = Field(
            default=None,
            description="Optional: Comma-separated keywords that force-save a memory "
            "even if blacklisted",
        )
        max_total_memories: int = Field(
            default=200,
            description="Maximum number of memories per user; prune oldest beyond this",
        )
        pruning_strategy: Literal["fifo", "least_relevant"] = Field(
            default="fifo",
            description="Strategy for pruning memories when max_total_memories is "
            "exceeded: 'fifo' (oldest first) or 'least_relevant' (lowest relevance "
            "to current message first).",
        )
        min_memory_length: int = Field(
            default=8,
            description="Minimum length of memory content to be saved",
        )
        min_confidence_threshold: float = Field(
            default=0.5,
            description="Minimum confidence score (0-1) required for an extracted "
            "memory to be saved. Scores below this are discarded.",
        )
        recent_messages_n: int = Field(
            default=5,
            description="Number of recent user messages to include in extraction "
            "prompt context",
        )
        save_relevance_threshold: float = Field(
            default=0.8,
            description="Minimum relevance score (based on relevance calculation "
            "method) to save a memory",
        )
        max_injected_memory_length: int = Field(
            default=300,
            description="Maximum length of each injected memory snippet",
        )

        # Generic LLM Provider Configuration
        llm_provider_type: Literal["ollama", "openai_compatible"] = Field(
            default="ollama",
            description="Type of LLM provider ('ollama' or 'openai_compatible')",
        )
        llm_model_name: str = Field(
            default="llama3:latest",
            description="Name of the LLM model to use (e.g., 'llama3:latest', 'gpt-4o')",
        )
        llm_api_endpoint_url: str = Field(
            default="http://host.docker.internal:11434/api/chat",
            description="API endpoint URL for the LLM provider (e.g., "
            "'http://host.docker.internal:11434/api/chat', "
            "'https://api.openai.com/v1/chat/completions')",
        )
        llm_api_key: Optional[str] = Field(
            default=None,
            description="API Key for the LLM provider (required if type is "
            "'openai_compatible')",
        )
        llm_timeout: int = Field(
            default=DEFAULT_LLM_TIMEOUT,
            description="Timeout in seconds for LLM API calls",
        )

        # Memory processing settings
        related_memories_n: int = Field(
            default=5,
            description="Number of related memories to consider",
        )
        relevance_threshold: float = Field(
            default=0.60,
            description="Minimum relevance score (0-1) for memories to be considered "
            "relevant for injection after scoring",
        )
        memory_threshold: float = Field(
            default=0.6,
            description="Threshold for similarity when comparing memories (0-1)",
        )
        vector_similarity_threshold: float = Field(
            default=0.60,
            description="Minimum cosine similarity for initial vector filtering (0-1)",
        )
        llm_skip_relevance_threshold: float = Field(
            default=0.93,
            description="If *all* vector-filtered memories have similarity >= this "
            "threshold, treat the vector score as final relevance and skip the "
            "additional LLM call.",
        )
        top_n_memories: int = Field(
            default=3,
            description="Number of top similar memories to pass to LLM",
        )
        cache_ttl_seconds: int = Field(
            default=86400,
            description="Cache time-to-live in seconds (default 24 hours)",
        )
        embedding_timeout: int = Field(
            default=30,
            description="Timeout in seconds for embedding API calls",
        )
        use_llm_for_relevance: bool = Field(
            default=False,
            description="Use LLM call for final relevance scoring (if False, relies "
            "solely on vector similarity + relevance_threshold)",
        )
        deduplicate_memories: bool = Field(
            default=True,
            description="Prevent storing duplicate or very similar memories",
        )
        use_embeddings_for_deduplication: bool = Field(
            default=True,
            description="Use embedding-based similarity for more accurate semantic "
            "duplicate detection (if False, uses text-based similarity)",
        )
        embedding_similarity_threshold: float = Field(
            default=0.75,
            description="Threshold (0-1) for considering two memories duplicates "
            "when using embedding similarity.",
        )
        similarity_threshold: float = Field(
            default=0.95,
            description="Threshold for detecting similar memories (0-1) using text "
            "or embeddings",
        )
        timezone: str = Field(
            default="Asia/Dubai",
            description="Timezone for date/time processing (e.g., 'America/New_York', "
            "'Europe/London')",
        )
        show_status: bool = Field(
            default=True, description="Show memory operations status in chat"
        )
        show_memories: bool = Field(
            default=True, description="Show relevant memories in context"
        )
        memory_format: Literal["bullet", "paragraph", "numbered"] = Field(
            default="bullet", description="Format for displaying memories in context"
        )
        enable_identity_memories: bool = Field(
            default=True,
            description="Enable collecting Basic Identity information (age, gender, "
            "location, etc.)",
        )
        enable_behavior_memories: bool = Field(
            default=True,
            description="Enable collecting Behavior information (interests, habits, "
            "etc.)",
        )
        enable_preference_memories: bool = Field(
            default=True,
            description="Enable collecting Preference information (likes, dislikes, "
            "etc.)",
        )
        enable_goal_memories: bool = Field(
            default=True,
            description="Enable collecting Goal information (aspirations, targets, "
            "etc.)",
        )
        enable_relationship_memories: bool = Field(
            default=True,
            description="Enable collecting Relationship information (friends, family, "
            "etc.)",
        )
        enable_possession_memories: bool = Field(
            default=True,
            description="Enable collecting Possession information (things owned or "
            "desired)",
        )
        max_retries: int = Field(
            default=2, description="Maximum number of retries for API calls"
        )
        retry_delay: float = Field(
            default=1.0, description="Delay between retries (seconds)"
        )

        # Prompts
        memory_identification_prompt: str = Field(
            default="""You are an automated JSON data extraction system. Your ONLY function is to identify user-specific, persistent facts, preferences, goals, relationships, or interests from the user's messages and output them STRICTLY as a JSON array of operations.

**ABSOLUTE OUTPUT REQUIREMENT: FAILURE TO COMPLY WILL BREAK THE SYSTEM.**
1.  Your **ENTIRE** response **MUST** be **ONLY** a valid JSON array starting with `[` and ending with `]`.
2.  **NO EXTRA TEXT**: Do **NOT** include **ANY** text, explanations, greetings, apologies, notes, or markdown formatting (like ```json) before or after the JSON array.
3.  **ARRAY ALWAYS**: Even if you find only one memory, it **MUST** be enclosed in an array: `[{"operation": ...}]`. Do **NOT** output a single JSON object `{...}`.
4.  **EMPTY ARRAY**: If NO relevant user-specific memories are found, output **ONLY** an empty JSON array: `[]`.

**JSON OBJECT STRUCTURE (Each element in the array):**
*   Each element **MUST** be a JSON object: `{"operation": "NEW", "content": "...", "tags": ["..."], "memory_bank": "...", "confidence": float}`
*   **confidence**: You **MUST** include a confidence score (float between 0.0 and 1.0) indicating certainty that the extracted text is a persistent user fact/preference. High confidence (0.8-1.0) for direct statements, lower (0.5-0.7) for inferences or less certain preferences.
*   **memory_bank**: You **MUST** include a `memory_bank` field, choosing from: "General", "Personal", "Work". Default to "General" if unsure.
*   **tags**: You **MUST** include a `tags` field with a list of relevant tags from: ["identity", "behavior", "preference", "goal", "relationship", "possession"].

**INFORMATION TO EXTRACT (User-Specific ONLY):**
*   **Explicit Preferences/Statements:** User states "I love X", "My favorite is Y", "I enjoy Z". Extract these verbatim with high confidence.
*   **Identity:** Name, location, age, profession, etc. (high confidence)
*   **Goals:** Aspirations, plans (medium/high confidence depending on certainty).
*   **Relationships:** Mentions of family, friends, colleagues (high confidence).
*   **Possessions:** Things owned or desired (medium/high confidence).
*   **Behaviors/Interests:** Topics the user discusses or asks about (implying interest - medium confidence).

**RULES (Reiteration - Critical):**
+1. **JSON ARRAY ONLY**: `[`...`]` - Nothing else!
+2. **CONFIDENCE REQUIRED**: Every object needs a `"confidence": float` field.
+3. **MEMORY BANK REQUIRED**: Every object needs a `"memory_bank": "..."` field.
+4. **TAGS REQUIRED**: Every object needs a `"tags": [...]` field.
+5. **USER INFO ONLY**: Discard trivia, questions *to* the AI, temporary thoughts.

**FAILURE EXAMPLES (DO NOT DO THIS):**
*   `Okay, here is the JSON: [...]` <-- INVALID (extra text)
*   ` ```json
[{"operation": ...}]
``` ` <-- INVALID (markdown)
*   `{"memories": [...]}` <-- INVALID (not an array)
*   `{"operation": ...}` <-- INVALID (not in an array)
*   `[{"operation": ..., "content": ..., "tags": [...]}]` <-- INVALID (missing confidence/bank)

**GOOD EXAMPLE OUTPUT (Strictly adhere to this):**
```
[
  {
    "operation": "NEW",
    "content": "User has been a software engineer for 8 years",
    "tags": ["identity", "behavior"],
    "memory_bank": "Work",
    "confidence": 0.95
  },
  {
    "operation": "NEW",
    "content": "User has a cat named Whiskers",
    "tags": ["relationship", "possession"],
    "memory_bank": "Personal",
    "confidence": 0.9
  },
  {
    "operation": "NEW",
    "content": "User prefers working remotely",
    "tags": ["preference", "behavior"],
    "memory_bank": "Work",
    "confidence": 0.7
  },
  {
    "operation": "NEW",
    "content": "User's favorite book might be The Hitchhiker's Guide to the Galaxy",
    "tags": ["preference"],
    "memory_bank": "Personal",
    "confidence": 0.6
  }
]
```

Analyze the following user message(s) and provide **ONLY** the JSON array output. Double-check your response starts with `[` and ends with `]` and contains **NO** other text whatsoever.""",
            description="System prompt for memory identification",
        )
        memory_relevance_prompt: str = Field(
            default="""You are a memory retrieval assistant. Your task is to determine which memories are relevant to the current context of a conversation.

IMPORTANT: **Do NOT mark general knowledge, trivia, or unrelated facts as relevant.** Only user-specific, persistent information should be rated highly.

Given the current user message and a set of memories, rate each memory's relevance on a scale from 0 to 1, where:
- 0 means completely irrelevant
- 1 means highly relevant and directly applicable

Consider:
- Explicit mentions in the user message
- Implicit connections to the user's personal info, preferences, goals, or relationships
- Potential usefulness for answering questions **about the user**
- Recency and importance of the memory

Examples:
- "User likes coffee" -> likely relevant if coffee is mentioned
- "World War II started in 1939" -> **irrelevant trivia, rate near 0**
- "User's friend is named Sarah" -> relevant if friend is mentioned

Return your analysis as a JSON array with each memory's content, ID, and relevance score.
Example: [{"memory": "User likes coffee", "id": "123", "relevance": 0.8}]

Your output must be valid JSON only. No additional text.""",
            description="System prompt for memory relevance assessment",
        )
        memory_merge_prompt: str = Field(
            default="""You are a memory consolidation assistant. When given sets of memories, you merge similar or related memories while preserving all important information.

IMPORTANT: **Do NOT merge general knowledge, trivia, or unrelated facts.** Only merge user-specific, persistent information.

Rules for merging:
1. If two memories contradict, keep the newer information
2. Combine complementary information into a single comprehensive memory
3. Maintain the most specific details when merging
4. If two memories are distinct enough, keep them separate
5. Remove duplicate memories

Return your result as a JSON array of strings, with each string being a merged memory.
Your output must be valid JSON only. No additional text.""",
            description="System prompt for merging memories",
        )

        # Memory Bank Config
        allowed_memory_banks: List[str] = Field(
            default=["General", "Personal", "Work"],
            description="List of allowed memory bank names for categorization.",
        )
        default_memory_bank: str = Field(
            default="General",
            description="Default memory bank assigned when LLM omits or supplies an "
            "invalid bank.",
        )

        # Validators
        @field_validator(
            "max_retries",
            "cache_ttl_seconds",
        )
        def check_non_negative_int(cls, v, info):
            if not isinstance(v, int) or v < 0:
                raise ValueError("%s must be a non-negative integer" % info.field_name)
            return v

        @field_validator(
            "summarization_interval",
            "error_logging_interval",
            "vector_cleanup_interval",
            "max_total_memories",
            "min_memory_length",
            "recent_messages_n",
            "related_memories_n",
            "top_n_memories",
            "max_injected_memory_length",
            "summarization_min_cluster_size",
            "summarization_max_cluster_size",
            "llm_timeout",
            "embedding_timeout",
        )
        def check_positive_int(cls, v, info):
            if not isinstance(v, int) or v <= 0:
                raise ValueError("%s must be a positive integer" % info.field_name)
            return v

        @field_validator(
            "save_relevance_threshold",
            "relevance_threshold",
            "memory_threshold",
            "vector_similarity_threshold",
            "similarity_threshold",
            "summarization_similarity_threshold",
            "llm_skip_relevance_threshold",
            "embedding_similarity_threshold",
            "min_confidence_threshold",
            check_fields=False,
        )
        def check_threshold_float(cls, v, info):
            if not (0.0 <= v <= 1.0):
                raise ValueError(
                    "%s must be between 0.0 and 1.0. Received: %s"
                    % (info.field_name, v)
                )
            return v

        @field_validator("retry_delay")
        def check_non_negative_float(cls, v, info):
            if not isinstance(v, (int, float)) or v < 0.0:
                raise ValueError("%s must be a non-negative float" % info.field_name)
            return float(v)

        @field_validator("timezone")
        def check_valid_timezone(cls, v: str) -> str:
            try:
                ZoneInfo(v)
            except (ZoneInfoNotFoundError, KeyError) as e:
                raise ValueError("Invalid timezone '%s': %s" % (v, e)) from e
            return v

        @model_validator(mode="after")
        def check_llm_config(self):
            if self.llm_provider_type == "openai_compatible" and not self.llm_api_key:
                raise ValueError(
                    "API Key is required when llm_provider_type is 'openai_compatible'"
                )
            return self

        @field_validator("allowed_memory_banks", check_fields=False)
        def check_allowed_memory_banks(cls, v):
            if not isinstance(v, list) or not v or v == [""]:
                return cls.model_fields["allowed_memory_banks"].default
            cleaned_list = [str(item).strip() for item in v if str(item).strip()]
            if not cleaned_list:
                return cls.model_fields["allowed_memory_banks"].default
            return cleaned_list

        @model_validator(mode="after")
        def check_embedding_config(self):
            if self.embedding_provider_type == "openai_compatible":
                if not self.embedding_api_key:
                    raise ValueError(
                        "API Key required for openai_compatible embedding provider"
                    )
            return self

    class UserValves(BaseModel):
        enabled: bool = Field(
            default=True, description="Enable or disable the memory function"
        )
        show_status: bool = Field(
            default=True, description="Show memory processing status updates"
        )
        timezone: str = Field(
            default="",
            description="User's timezone (overrides global setting if provided)",
        )

    # --------------------------------------------------------------------------
    # Main Filter Initialization
    # --------------------------------------------------------------------------

    def __init__(self):
        logger.info("Initializing Adaptive Memory Filter v%s", __version__)
        self.valves = self.Valves()
        self.error_manager = ErrorManager()
        # Pass a lambda to always get the current valves state
        self.embedding_manager = EmbeddingManager(
            lambda: self.valves, self.error_manager
        )
        self.task_manager = TaskManager(self)
        # Create pipeline once and reuse - pass getter to avoid stale valves
        self.pipeline = MemoryPipeline(
            lambda: self.valves, self.embedding_manager, self.error_manager
        )

        # Initialize internal state
        self._last_body = {}
        self.seen_users = set()  # Track active users for background tasks
        self.notification_queue = deque(maxlen=NOTIFICATION_QUEUE_MAXLEN)
        self._tasks_started = False
        self._valve_hash = None  # Track valve changes
        self._task_lock_obj = None  # Lazy initialization for async lock
        self._restart_task = None
        self._llm_session = None  # Reusable LLM session
        self._llm_session_lock_obj = None  # Lock for LLM session creation

        logger.info("Adaptive Memory Filter v%s initialized", __version__)

    @property
    def _task_lock(self) -> asyncio.Lock:
        """Lazy initialization of async lock to avoid wrong event loop binding.

        NOTE: This is thread-safe under asyncio's cooperative scheduling model.
        Python's async/await only yields at await points, so the check-then-set
        pattern cannot race within a single event loop.
        """
        if self._task_lock_obj is None:
            self._task_lock_obj = asyncio.Lock()
        return self._task_lock_obj

    @property
    def _llm_session_lock(self) -> asyncio.Lock:
        """Lazy initialization of async lock for LLM session.

        NOTE: This is thread-safe under asyncio's cooperative scheduling model.
        Python's async/await only yields at await points, so the check-then-set
        pattern cannot race within a single event loop.
        """
        if self._llm_session_lock_obj is None:
            self._llm_session_lock_obj = asyncio.Lock()
        return self._llm_session_lock_obj

    async def _ensure_llm_session(self):
        """Ensure LLM session exists (async-safe)."""
        async with self._llm_session_lock:
            if not self._llm_session or self._llm_session.closed:
                self._llm_session = aiohttp.ClientSession()

    async def _check_and_handle_valve_changes(self):
        """Detect if valves have changed and restart tasks if needed."""
        # Hash important valve settings that affect background tasks
        valve_str = "%s_%s_%s_%s_%s_%s_%s" % (
            self.valves.enable_summarization_task,
            self.valves.summarization_interval,
            self.valves.enable_error_logging_task,
            self.valves.error_logging_interval,
            self.valves.enable_vector_cleanup_task,
            self.valves.vector_cleanup_interval,
            self.valves.cache_ttl_seconds,
        )
        new_hash = hashlib.md5(valve_str.encode(), usedforsecurity=False).hexdigest()

        async with self._task_lock:
            if self._valve_hash is None:
                self._valve_hash = new_hash
                return False

            if new_hash != self._valve_hash:
                logger.info("Valve changes detected! Restarting background tasks...")
                self._valve_hash = new_hash
                if self._tasks_started:
                    # Create managed task with error callback
                    self._restart_task = asyncio.create_task(self._restart_tasks())

                    def _log_restart_exception(task):
                        try:
                            task.result()
                        except asyncio.CancelledError:
                            pass  # Expected when task is cancelled
                        except Exception as e:
                            logger.exception("Background restart task failed: %s", e)

                    self._restart_task.add_done_callback(_log_restart_exception)
                return True
        return False

    async def _restart_tasks(self):
        """Restart background tasks with new valve settings."""
        # Cancel existing restart task if running (but not ourselves)
        current_task = asyncio.current_task()
        if (
            hasattr(self, "_restart_task")
            and self._restart_task
            and not self._restart_task.done()
            and self._restart_task is not current_task
        ):
            self._restart_task.cancel()
            try:
                await self._restart_task
            except asyncio.CancelledError:
                pass

        # Stop tasks outside the lock to avoid deadlock with cleanup()
        await self.task_manager.stop_tasks()

        async with self._task_lock:
            self._tasks_started = False
            self.task_manager.start_tasks()
            self._tasks_started = True
            logger.info("Background tasks restarted with new valve values")

    async def cleanup(self):
        """Clean up resources."""
        async with self._task_lock:
            await self.task_manager.stop_tasks()
        await self.embedding_manager.cleanup()
        if self._llm_session:
            await self._llm_session.close()
            self._llm_session = None

    # --------------------------------------------------------------------------
    # Helper: LLM Query Wrapper
    # --------------------------------------------------------------------------
    async def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Unified LLM query method with retries and metrics.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM

        Returns:
            LLM response text or None if failed
        """
        valves = self.valves

        # Lazily create and reuse LLM session (async-safe)
        await self._ensure_llm_session()

        for attempt in range(valves.max_retries + 1):
            try:
                url = valves.llm_api_endpoint_url
                headers = {"Content-Type": "application/json"}
                if valves.llm_api_key:
                    headers["Authorization"] = "Bearer %s" % valves.llm_api_key

                payload = {
                    "model": valves.llm_model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                }

                # Use configurable timeout from valves
                async with self._llm_session.post(
                    url, json=payload, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=valves.llm_timeout)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        # Safe extraction logic for different response formats
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            return choice.get("message", {}).get("content")
                        elif "message" in data:
                            return data["message"].get("content")
                        logger.warning("Unexpected LLM response format: %s", list(data.keys()))
                        return None
                    elif resp.status >= 500:
                        # Server error - retry
                        raise aiohttp.ClientError("Server error: %d" % resp.status)
                    else:
                        # Client error (4xx) - don't retry
                        logger.warning(
                            "LLM API client error %d, not retrying", resp.status
                        )
                        return None

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # Re-create session if closed, then retry
                if self._llm_session and self._llm_session.closed:
                    await self._ensure_llm_session()
                if attempt < valves.max_retries:
                    logger.warning(
                        "LLM query attempt %d/%d failed, retrying in %ss: %s",
                        attempt + 1,
                        valves.max_retries + 1,
                        valves.retry_delay,
                        e,
                    )
                    await asyncio.sleep(valves.retry_delay)
                else:
                    logger.exception(
                        "LLM Query failed after %d attempts: %s",
                        valves.max_retries + 1,
                        e,
                    )
                    await self.error_manager.increment("llm_call_errors")
            except Exception as e:
                logger.exception("Unexpected LLM query error: %s", e)
                await self.error_manager.increment("llm_call_errors")
                break

        return None

    def _get_user_valves(self, raw_user: dict) -> "Filter.UserValves":
        """Extract and validate user valves from raw user data.

        Args:
            raw_user: User dictionary containing valves

        Returns:
            Validated UserValves instance
        """
        raw_valves = raw_user.get("valves", {})
        if hasattr(raw_valves, "model_dump"):
            valves_dict = raw_valves.model_dump()
        elif hasattr(raw_valves, "dict"):
            valves_dict = raw_valves.dict()
        elif isinstance(raw_valves, dict):
            valves_dict = raw_valves
        elif isinstance(raw_valves, self.UserValves):
            return raw_valves
        else:
            valves_dict = {}
        try:
            return self.UserValves(**valves_dict)
        except Exception:
            return self.UserValves()

    # --------------------------------------------------------------------------
    # Core Pipeline: Inlet (Incoming Message)
    # --------------------------------------------------------------------------
    async def inlet(
        self, body: Dict[str, Any], __event_emitter__=None, __user__=None
    ) -> Dict[str, Any]:
        """Process incoming message: Identify user, inject context memories.

        Args:
            body: Request body containing messages
            __event_emitter__: Optional event emitter for status updates
            __user__: User information dict

        Returns:
            Modified body with injected memories
        """
        if not __user__ or not body.get("messages"):
            return body

        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            return body

        # Start tasks with proper locking
        if not self._tasks_started:
            async with self._task_lock:
                if not self._tasks_started:
                    self.task_manager.start_tasks()
                    self._tasks_started = True

        # Check if valves have changed and restart tasks if needed
        await self._check_and_handle_valve_changes()

        user_id = __user__["id"]
        self.seen_users.add(user_id)  # Track active user
        messages = body["messages"]
        last_message_content = messages[-1]["content"]

        # Handle multimodal content (list of dicts)
        if isinstance(last_message_content, list):
            last_message = " ".join(
                [
                    m.get("text", "")
                    for m in last_message_content
                    if isinstance(m, dict) and m.get("type") == "text"
                ]
            ).strip()
        else:
            last_message = last_message_content

        # Skip command processing
        if last_message.startswith("/"):
            logger.info("Skipping memory processing for command")
            return body

        if not last_message:
            logger.debug("Skipping memory processing for empty message.")
            return body

        # 1. Retrieve all memories
        try:
            # This gets generic memory objects (wrapped in to_thread)
            all_memories = []
            if Memories is not None:
                all_memories = await asyncio.to_thread(
                    Memories.get_memories_by_user_id, user_id
                )
        except Exception as e:
            logger.error("Failed to fetch memories: %s", e)
            all_memories = []

        # 2. Filter relevant memories
        relevant_memories = []
        if all_memories:
            relevant_memories = await self.pipeline.get_relevant_memories(
                last_message, user_id, all_memories
            )
            logger.info(
                "Memory retrieval: found %d relevant memories from %d total memories",
                len(relevant_memories),
                len(all_memories),
            )
        else:
            logger.debug("Memory retrieval: no existing memories found for user %s", user_id)

        # 3. Inject into system prompt
        if relevant_memories:
            # Strip internal metadata annotations ([Tags: ...], [Memory Bank: ...],
            # [Confidence: ...]) before injection so thinking-capable models (e.g.
            # Gemini Flash) don't treat schema markers as competing instructions and
            # enter an infinite reasoning loop.
            context_text = "User Memories:\n" + "\n".join(
                [
                    "- %s" % self.pipeline._extract_raw_content(m.content)
                    for m in relevant_memories
                ]
            )

            # Validate system message before injection
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] += "\n\n%s" % context_text
            else:
                # Insert a new system message if none exists
                messages.insert(0, {"role": "system", "content": context_text})

            # Show status if enabled
            if user_valves.show_status:
                # 1. Recall Notifications
                count = len(relevant_memories)
                if count > 0:
                    suffix = "memory" if count == 1 else "memories"
                    status_dict = {
                        "type": "status",
                        "data": {
                            "description": "🧠 Recalled %d %s." % (count, suffix),
                            "done": True,
                        },
                    }
                    if __event_emitter__:
                        await __event_emitter__(status_dict)

                # 2. Background Notifications (queued)
                while self.notification_queue:
                    msg = self.notification_queue.popleft()
                    bg_status_dict = {
                        "type": "status",
                        "data": {"description": "🧹 %s" % msg, "done": True},
                    }
                    if __event_emitter__:
                        await __event_emitter__(bg_status_dict)

        return body

    # --------------------------------------------------------------------------
    # Core Pipeline: Outlet (Response Processing)
    # --------------------------------------------------------------------------
    async def outlet(
        self, body: Dict[str, Any], __event_emitter__=None, __user__=None
    ) -> Dict[str, Any]:
        """Process outgoing response: Extract memories, update status.

        Args:
            body: Request body containing messages
            __event_emitter__: Optional event emitter for status updates
            __user__: User information dict

        Returns:
            Unmodified body
        """
        if not __user__ or not body.get("messages"):
            return body

        user_valves = self._get_user_valves(__user__)
        if not user_valves.enabled:
            return body

        user_id = __user__["id"]
        messages = body["messages"]

        # Get last user message with multimodal support
        user_message = ""
        for m in reversed(messages):
            if m["role"] == "user":
                raw_content = m["content"]
                if isinstance(raw_content, list):
                    user_message = " ".join(
                        part.get("text", "")
                        for part in raw_content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ).strip()
                else:
                    user_message = raw_content
                break

        # Identify Memories
        if user_message:
            # Pass our _query_llm as callback
            ops = await self.pipeline.identify_memories(
                user_message,
                context_memories=[],
                query_llm_func=self._query_llm,
            )
            logger.info(
                "Memory extraction: identified %d potential memories from user message",
                len(ops),
            )

            success_ops = []
            if ops:
                # Process Operations (Save/Delete)
                success_ops = await self.pipeline.process_memory_operations(
                    ops, user_id
                )

            if len(success_ops) > 0:
                logger.info(
                    "Memory operations: saved %d new memories (skipped %d duplicates)",
                    len(success_ops),
                    len(ops) - len(success_ops),
                )
            elif len(ops) > 0:
                logger.info(
                    "Memory operations: all %d identified memories were duplicates, "
                    "none saved",
                    len(ops),
                )
            else:
                logger.debug("Memory operations: no memories identified from user message")

            # Show status if enabled
            if user_valves.show_status:
                count = len(success_ops)
                if count > 0:
                    suffix = "memory" if count == 1 else "memories"
                    description = "🧠 Saved %d %s." % (count, suffix)
                else:
                    description = "No memories saved."

                status_dict = {
                    "type": "status",
                    "data": {"description": description, "done": True},
                }
                if __event_emitter__:
                    await __event_emitter__(status_dict)
                else:
                    logger.warning("Outlet: No event emitter available for status.")

        return body

    # ... Placeholder for other required methods (referenced by TaskManager) ...
    async def _summarize_old_memories_loop(self):
        """Background task for summarization."""
        logger.info(
            "Summarization background task launched with interval: %d seconds",
            self.valves.summarization_interval,
        )
        while True:
            try:
                # Always get the current valve value in case it changed
                interval = self.valves.summarization_interval
                await asyncio.sleep(interval)
                logger.info(
                    "Summarization task running. Active users: %d, enabled: %s",
                    len(self.seen_users),
                    self.valves.enable_summarization_task,
                )

                if self.valves.enable_summarization_task and self.seen_users:
                    logger.info("Background summarization: starting scan...")

                    # Copy set to avoid size change during iteration
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        try:
                            logger.info(
                                "Background summarization: processing user %s", user_id
                            )
                            # Use _query_llm as callback
                            result_msg = await self.pipeline.cluster_and_summarize(
                                user_id, self._query_llm
                            )
                            if result_msg and isinstance(result_msg, str):
                                self.notification_queue.append(result_msg)
                                logger.info("Background summarization: %s", result_msg)
                            else:
                                logger.debug(
                                    "Background summarization: no clusters found for user %s",
                                    user_id,
                                )
                        except Exception as u_err:
                            logger.exception(
                                "Background summarization error for user %s: %s",
                                user_id,
                                u_err,
                            )

                    logger.info("Background summarization: cycle complete")
                else:
                    logger.debug(
                        "Background summarization: skipped (enabled: %s, users: %d)",
                        self.valves.enable_summarization_task,
                        len(self.seen_users),
                    )

            except asyncio.CancelledError:
                logger.info("Summarization task cancelled")
                break
            except Exception as e:
                logger.exception("Summarization task error: %s", e)
                await asyncio.sleep(DEFAULT_BACKGROUND_ERROR_SLEEP)

    async def _log_error_counters_loop(self):
        """Periodically log error counters."""
        try:
            while True:
                await asyncio.sleep(self.valves.error_logging_interval)
                counters = self.error_manager.get_counters()
                if any(v > 0 for v in counters.values()):
                    logger.warning("Error Counters (non-zero): %s", counters)
                else:
                    logger.debug("Error Counters (all zero): %s", counters)
        except asyncio.CancelledError:
            logger.info("Error logging task cancelled")
        except Exception as e:
            logger.exception("Error in error logging loop: %s", e)

    async def cleanup_orphaned_vectors(self, user_id: str) -> Dict[str, Union[int, str]]:
        """
        Audit and clean up orphaned vector embeddings.

        Args:
            user_id: User ID to clean up vectors for

        Returns:
            Dict with:
            - db_memories: count of memories in database
            - orphans_deleted: count of orphaned vectors removed
            - error: (optional) error message if cleanup failed
        """
        if not VECTOR_DB_CLIENT:
            logger.warning("Vector DB not available - cannot cleanup orphaned vectors")
            return {"error": "Vector DB not available", "orphans_deleted": 0}

        try:
            # Get all memory IDs from database
            db_memories = []
            if Memories is not None:
                db_memories = await asyncio.to_thread(
                    Memories.get_memories_by_user_id, user_id
                )
            if db_memories is None:
                return {
                    "error": "Failed to retrieve memories from database",
                    "orphans_deleted": 0,
                }
            valid_ids = {str(m.id) for m in db_memories}

            collection_name = "user-memory-%s" % user_id

            # Get all vector IDs - method depends on vector DB implementation
            try:
                # Try to get all items from collection (run in thread to avoid blocking)
                result = await asyncio.to_thread(
                    VECTOR_DB_CLIENT.get, collection_name=collection_name
                )
                if result and "ids" in result:
                    vector_ids = result["ids"]
                else:
                    logger.warning(
                        "Unable to retrieve vector IDs for cleanup - collection may "
                        "not exist"
                    )
                    return {"db_memories": len(valid_ids), "orphans_deleted": 0}
            except Exception as e:
                logger.error("Failed to retrieve vector IDs: %s", e)
                return {"error": str(e), "orphans_deleted": 0}

            # Find orphans (vectors without corresponding database entry)
            orphaned_ids = [vid for vid in vector_ids if vid not in valid_ids]

            # Delete orphans
            if orphaned_ids:
                try:
                    await asyncio.to_thread(
                        VECTOR_DB_CLIENT.delete,
                        collection_name=collection_name,
                        ids=orphaned_ids,
                    )
                    logger.info(
                        "Deleted %d orphaned vectors for user %s",
                        len(orphaned_ids),
                        user_id,
                    )
                except Exception as del_err:
                    logger.error("Failed to delete orphaned vectors: %s", del_err)
                    return {
                        "db_memories": len(valid_ids),
                        "orphans_found": len(orphaned_ids),
                        "orphans_deleted": 0,
                        "error": str(del_err),
                    }
            else:
                logger.info("No orphaned vectors found for user %s", user_id)

            return {
                "db_memories": len(valid_ids),
                "vector_count": len(vector_ids),
                "orphans_deleted": len(orphaned_ids),
            }

        except Exception as e:
            logger.exception("Error during vector cleanup: %s", e)
            return {"error": str(e), "orphans_deleted": 0}

    async def _cleanup_vectors_loop(self):
        """Background task for cleaning up orphaned vectors."""
        logger.info(
            "Vector cleanup background task launched with interval: %d seconds",
            self.valves.vector_cleanup_interval,
        )
        while True:
            try:
                await asyncio.sleep(self.valves.vector_cleanup_interval)
                logger.info(
                    "Vector cleanup task running. Active users: %d, enabled: %s",
                    len(self.seen_users),
                    self.valves.enable_vector_cleanup_task,
                )

                if self.valves.enable_vector_cleanup_task and self.seen_users:
                    logger.info("Background vector cleanup: starting scan...")

                    # Copy set to avoid size change during iteration
                    active_users = list(self.seen_users)
                    for user_id in active_users:
                        try:
                            logger.info(
                                "Background vector cleanup: processing user %s", user_id
                            )
                            result = await self.cleanup_orphaned_vectors(user_id)

                            if "orphans_deleted" in result and isinstance(result["orphans_deleted"], int) and result["orphans_deleted"] > 0:
                                msg = (
                                    "Cleaned up %d orphaned vectors for user %s"
                                    % (result["orphans_deleted"], user_id)
                                )
                                self.notification_queue.append(msg)
                                logger.info("Background vector cleanup: %s", msg)
                            else:
                                logger.debug(
                                    "Background vector cleanup: no orphans found for user %s",
                                    user_id,
                                )
                        except Exception as u_err:
                            logger.exception(
                                "Background vector cleanup error for user %s: %s",
                                user_id,
                                u_err,
                            )

                    logger.info("Background vector cleanup: cycle complete")
                else:
                    logger.debug(
                        "Background vector cleanup: skipped (enabled: %s, users: %d)",
                        self.valves.enable_vector_cleanup_task,
                        len(self.seen_users),
                    )

            except asyncio.CancelledError:
                logger.info("Vector cleanup task cancelled")
                break
            except Exception as e:
                logger.exception("Vector cleanup task error: %s", e)
                await asyncio.sleep(DEFAULT_BACKGROUND_ERROR_SLEEP)
