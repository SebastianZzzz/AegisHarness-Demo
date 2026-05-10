import sys
import threading
import time
from collections import OrderedDict
from collections.abc import MutableMapping, KeysView, ValuesView, ItemsView
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterator, Optional, Tuple, TypeVar

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class _Entry(Generic[K, V]):
    key: K
    value: V
    expiry: Optional[float]
    size: int


class TTLDict(MutableMapping[K, V], Generic[K, V]):
    """
    A thread-safe TTL-enabled dictionary with LRU eviction semantics.

    Features:
    - Monotonic clock (time.monotonic) for all TTL expiry calculations.
    - Thread-safe public API via a re-entrant lock.
    - Expire-on-access and prune-on-iteration behavior.
    - Optional background cleanup thread for expired entries.
    - Proper memory accounting with a user-provided size function (defaults to sys.getsizeof).
    - Key immutability/hashability enforcement (keys must be hashable).
    - LRU eviction when capacity is exceeded (evicts oldest items first).
    - Full MutableMapping API compliance with non-expired item views.
    - Iteration reflects only valid (non-expired) items in LRU order (oldest to newest).
    - max_size <= 0 means an effectively empty cache (no storage).
    """

    def __init__(
        self,
        max_size: int = 1024,
        default_ttl: Optional[float] = None,
        size_func: Optional[Callable[[K, V], int]] = None,
        cleanup_interval: Optional[float] = None,
    ) -> None:
        """
        Initialize the TTLDict.

        Args:
            max_size: Maximum memory usage allowed by the cache (in bytes). If <= 0, cache stores nothing.
            default_ttl: Default TTL in seconds for new items. None means no expiry by default.
            size_func: Optional function to compute the size (in bytes) of an item (key, value).
                       If None, a conservative default based on sys.getsizeof is used.
            cleanup_interval: Optional background cleanup interval in seconds. If provided and > 0,
                              a daemon thread will periodically prune expired items.
        """
        self._max_size: int = int(max_size)
        self._default_ttl: Optional[float] = default_ttl
        self._size_func: Callable[[K, V], int] = size_func if size_func is not None else self._default_size
        self._entries: Dict[K, _Entry[K, V]] = {}
        self._order: "OrderedDict[K, None]" = OrderedDict()  # LRU order: oldest -> newest
        self._lock: threading.RLock = threading.RLock()
        self._total_size: int = 0

        self._cleanup_interval: Optional[float] = cleanup_interval
        self._stop_event: threading.Event = threading.Event()
        self._maintenance_thread: Optional[threading.Thread] = None
        if self._cleanup_interval is not None and self._cleanup_interval > 0:
            self._start_background_cleanup()

    # ------------------ Internal helpers ------------------ #

    @staticmethod
    def _default_size(key: Any, value: Any) -> int:
        """
        Default conservative size function using sys.getsizeof.
        """
        try:
            return sys.getsizeof(key) + sys.getsizeof(value)
        except Exception:
            return 0

    def _now(self) -> float:
        return time.monotonic()

    def _validate_key(self, key: K) -> None:
        """
        Validate key is hashable (immutable key requirement). Raises TypeError if not.
        """
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError(f"Keys must be hashable and immutable. {exc}")

    def _expire_and_prune(self) -> None:
        """
        Remove expired items based on monotonic time.
        """
        now = self._now()
        to_remove = [k for k in self._order.keys()
                     if self._entries.get(k) is not None and self._entries[k].expiry is not None and self._entries[k].expiry <= now]

        for key in to_remove:
            self._remove_entry(key)

    def _remove_entry(self, key: K) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._order.pop(key, None)
            self._total_size -= entry.size

    def _evict_oldest(self) -> Optional[K]:
        """
        Evict the oldest (least recently used) item. Returns the evicted key if any.
        """
        try:
            oldest_key, _ = self._order.popitem(last=False)
        except KeyError:
            return None
        self._remove_entry(oldest_key)
        return oldest_key

    def _ensure_within_capacity(self) -> None:
        """
        Evict items until total_size <= max_size (if max_size > 0).
        """
        if self._max_size <= 0:
            # No storage allowed; clear all
            self.clear()
            return

        while self._total_size > self._max_size:
            evicted = self._evict_oldest()
            if evicted is None:
                break  # nothing left to evict

    def _maybe_adjust_size_on_update(self, key: K, new_size: int) -> None:
        """
        Adjust total_size for an updated item and ensure capacity.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        delta = new_size - entry.size
        self._total_size += delta
        entry.size = new_size

    def _start_background_cleanup(self) -> None:
        self._stop_event.clear()
        self._maintenance_thread = threading.Thread(
            target=self._background_cleanup_loop,
            name="TTLDict-Cleanup",
            daemon=True
        )
        self._maintenance_thread.start()

    def _background_cleanup_loop(self) -> None:
        interval = max(0.001, float(self._cleanup_interval))
        while not self._stop_event.wait(interval):
            with self._lock:
                self._expire_and_prune()

    def stop_cleanup(self) -> None:
        """
        Stop the background cleanup thread gracefully.
        """
        self._stop_event.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=0.5)

    # ------------------ MutableMapping API ------------------ #

    def __setitem__(self, key: K, value: V) -> None:
        """
        Set an item with the default TTL. If max_size <= 0, the cache stores nothing.
        """
        with self._lock:
            self._validate_key(key)
            if self._max_size <= 0:
                # Treat as effectively empty; clear and skip storing
                self.clear()
                return

            ttl = self._default_ttl
            expiry = None
            if ttl is not None:
                expiry = self._now() + ttl

            size = self._size_func(key, value)

            existing = self._entries.get(key)
            if existing is not None:
                # Update existing item: adjust value, expiry, size, and recency
                existing.value = value
                existing.expiry = expiry
                self._order.move_to_end(key, last=True)
                self._maybe_adjust_size_on_update(key, size)
            else:
                # Ensure capacity before insertion
                self._entries[key] = _Entry(key=key, value=value, expiry=expiry, size=size)
                self._order[key] = None  # oldest to newest
                self._total_size += size

            # After adding/updating, prune expired (in case expiry is in the past) and evict if needed
            self._expire_and_prune()
            self._ensure_within_capacity()

    def __getitem__(self, key: K) -> V:
        with self._lock:
            self._expire_and_prune()
            entry = self._entries.get(key)
            if entry is None:
                raise KeyError(key)
            if entry.expiry is not None and entry.expiry <= self._now():
                # expired; remove and raise
                self._remove_entry(key)
                raise KeyError(key)
            # Update recency
            self._order.move_to_end(key, last=True)
            return entry.value

    def __delitem__(self, key: K) -> None:
        with self._lock:
            self._expire_and_prune()
            if key not in self._entries:
                raise KeyError(key)
            self._remove_entry(key)

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            self._expire_and_prune()
            # Snapshot of current keys in LRU order (oldest to newest)
            keys_snapshot = list(self._order.keys())
        for k in keys_snapshot:
            yield k

    def __len__(self) -> int:
        with self._lock:
            self._expire_and_prune()
            return len(self._entries)

    # Convenience API to align with common TTL caches

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        with self._lock:
            self._expire_and_prune()
            entry = self._entries.get(key)
            if entry is None:
                return default
            if entry.expiry is not None and entry.expiry <= self._now():
                self._remove_entry(key)
                return default
            self._order.move_to_end(key, last=True)
            return entry.value

    def set_with_ttl(self, key: K, value: V, ttl: Optional[float]) -> None:
        """
        Set an item with a specific TTL (in seconds). If None, use the default TTL.
        """
        with self._lock:
            self._validate_key(key)
            if self._max_size <= 0:
                self.clear()
                return

            expiry: Optional[float] = None
            if ttl is not None:
                expiry = self._now() + ttl

            size = self._size_func(key, value)
            existing = self._entries.get(key)
            if existing is not None:
                existing.value = value
                existing.expiry = expiry
                existing.size = size
                self._order.move_to_end(key, last=True)
                self._maybe_adjust_size_on_update(key, size)
            else:
                self._entries[key] = _Entry(key=key, value=value, expiry=expiry, size=size)
                self._order[key] = None
                self._total_size += size

            self._expire_and_prune()
            self._ensure_within_capacity()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._order.clear()
            self._total_size = 0

    def pop(self, key: K, default: Optional[V] = None) -> V:
        with self._lock:
            self._expire_and_prune()
            if key in self._entries:
                value = self._entries[key].value
                self._remove_entry(key)
                return value
            if default is not None:
                return default
            raise KeyError(key)

    def popitem(self) -> Tuple[K, V]:
        with self._lock:
            self._expire_and_prune()
            if not self._order:
                raise KeyError("dictionary is empty")
            oldest_key, _ = self._order.popitem(last=False)
            entry = self._entries.pop(oldest_key, None)
            if entry is None:
                raise KeyError(oldest_key)
            self._total_size -= entry.size
            return oldest_key, entry.value

    def update(self, *args, **kwargs) -> None:
        """
        Update the dictionary with another mapping or iterable of pairs.
        TTL for updated keys uses the default TTL unless TTL is provided via set_with_ttl.
        """
        if args:
            if len(args) > 1:
                raise TypeError("update() takes at most one positional argument ({} given)".format(len(args)))
            other = dict(args[0])
            for k, v in other.items():
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def keys(self) -> KeysView[K]:
        with self._lock:
            self._expire_and_prune()
            # Return a dynamic view-like object
            return KeysView(self)

    def values(self) -> ValuesView[V]:
        with self._lock:
            self._expire_and_prune()
            return ValuesView(self)

    def items(self) -> ItemsView[K, V]:
        with self._lock:
            self._expire_and_prune()
            return ItemsView(self)

    # Optional: representation
    def __repr__(self) -> str:
        with self._lock:
            self._expire_and_prune()
            items = []
            for k in self._order.keys():
                e = self._entries.get(k)
                if e is not None:
                    items.append(f"{k!r}: {e.value!r}")
            return f"{self.__class__.__name__}({{{', '.join(items)}}})"

    # ------------------ End of API ------------------ #

# End of TTLDict implementation