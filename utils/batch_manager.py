"""Queue-based batch manager for translation processing with optimized retry logic."""

import asyncio
import logging
from typing import Dict, List, Tuple, Callable, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Represents a single item in the batch queue."""
    key: str
    value: Any
    retry_count: int = 0


class BatchManager:
    """
    Queue-based batch manager that optimizes memory usage and ensures no omissions.
    
    Features:
    - Queue-based processing (no recursion, better heap management)
    - Automatic retry by pushing failed items back to queue
    - Configurable batch size
    - Progress tracking
    - Easy to expand and manage
    """
    
    def __init__(
        self,
        batch_size: int,
        max_retries: int = 0,  # 0 means infinite retries
        batch_builder: Optional[Callable[[List[BatchItem]], Any]] = None,
        on_success: Optional[Callable[[List[BatchItem], Any], None]] = None,
        on_failure: Optional[Callable[[List[BatchItem], Exception], None]] = None,
    ):
        """
        Initialize the batch manager.
        
        Args:
            batch_size: Maximum number of items per batch
            max_retries: Maximum retry count per item (0 = infinite)
            batch_builder: Optional function to build batch from items (e.g., for token limits)
            on_success: Optional callback when batch succeeds (item, result)
            on_failure: Optional callback when batch fails (item, exception)
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.batch_builder = batch_builder
        self.on_success = on_success
        self.on_failure = on_failure
        
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.stats = {
            'total_items': 0,
            'processed_items': 0,
            'failed_items': 0,
            'retry_count': 0,
        }
    
    async def add_items(self, items: List[Tuple[str, Any]]):
        """Add items to the processing queue."""
        for key, value in items:
            await self.queue.put(BatchItem(key=key, value=value))
            self.stats['total_items'] += 1
    
    async def add_item(self, key: str, value: Any):
        """Add a single item to the processing queue."""
        await self.queue.put(BatchItem(key=key, value=value))
        self.stats['total_items'] += 1
    
    def _build_batch(self, items: List[BatchItem]) -> Any:
        """
        Build a batch from items.
        If batch_builder is provided, use it; otherwise return items as-is.
        Returns either a list of BatchItems or a list of lists of BatchItems (for token-aware splitting).
        """
        if self.batch_builder:
            return self.batch_builder(items)
        return items
    
    async def process(
        self,
        processor: Callable[[Any], Any],
        concurrency: int = 1,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process all items in the queue using the provided processor function.
        
        Args:
            processor: Async function that processes a batch and returns results
            concurrency: Number of concurrent batch processors
            progress_callback: Optional callback for progress updates (called with item count)
        
        Returns:
            Dictionary mapping keys to processed results
        """
        if self.processing:
            raise RuntimeError("Batch manager is already processing")
        
        self.processing = True
        results: Dict[str, Any] = {}
        results_lock = asyncio.Lock()
        
        async def process_sub_batch(sub_batch: List[BatchItem]):
            """Process a single sub-batch independently."""
            if not sub_batch:
                return
            
            try:
                batch_results = await processor(sub_batch)
                
                # Handle success
                if self.on_success:
                    await self._call_callback(self.on_success, sub_batch, batch_results)
                
                # Store results
                async with results_lock:
                    if isinstance(batch_results, dict):
                        # Dictionary results (key -> value mapping)
                        for item in sub_batch:
                            if item.key in batch_results:
                                results[item.key] = batch_results[item.key]
                                self.stats['processed_items'] += 1
                            else:
                                # Missing key, put back in queue
                                await self._requeue_item(item)
                    elif isinstance(batch_results, list):
                        # List results (same order as sub_batch)
                        for item, result in zip(sub_batch, batch_results):
                            if result is not None:
                                results[item.key] = result
                                self.stats['processed_items'] += 1
                            else:
                                # None result, put back in queue
                                await self._requeue_item(item)
                    else:
                        # Single result or other format
                        logger.warning(f"Unexpected batch result type: {type(batch_results)}")
                        # Put all items back
                        for item in sub_batch:
                            await self._requeue_item(item)
                
                # Update progress
                if progress_callback:
                    progress_callback(len(sub_batch))
            
            except Exception as e:
                # Handle failure
                if self.on_failure:
                    await self._call_callback(self.on_failure, sub_batch, e)
                
                # Put failed items back in queue for retry
                for item in sub_batch:
                    await self._requeue_item(item)
                
                logger.warning(f"Sub-batch processing failed: {e}, requeuing {len(sub_batch)} items")
        
        async def process_batch_worker():
            """Worker that processes batches from the queue without blocking on slow batches."""
            active_tasks = set()
            
            try:
                while True:
                    # Clean up completed tasks
                    active_tasks = {t for t in active_tasks if not t.done()}
                    
                    # Get items for a batch (non-blocking if possible)
                    batch_items: List[BatchItem] = []
                    
                    # Try to get at least one item (with short timeout to avoid blocking)
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=0.1)
                        batch_items.append(item)
                    except asyncio.TimeoutError:
                        # No items available right now
                        # Check if we should exit: queue empty AND no active tasks
                        if self.queue.empty() and not active_tasks:
                            # Double-check with a slightly longer wait
                            try:
                                item = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                                # Item arrived, put it back and continue
                                await self.queue.put(item)
                            except asyncio.TimeoutError:
                                # Queue still empty and no active tasks - exit
                                break
                        # If we have active tasks, wait a bit for them to complete
                        if active_tasks:
                            await asyncio.sleep(0.01)
                        continue
                    
                    # Try to get more items up to batch_size (non-blocking)
                    while len(batch_items) < self.batch_size:
                        try:
                            item = self.queue.get_nowait()
                            batch_items.append(item)
                        except asyncio.QueueEmpty:
                            break
                    
                    if not batch_items:
                        continue
                    
                    # Build batch (may return single batch or list of sub-batches)
                    built_batch = self._build_batch(batch_items)
                    
                    # Normalize to list of batches (batch_builder may return multiple sub-batches)
                    if not built_batch:
                        # Empty batch, skip
                        continue
                    elif isinstance(built_batch, list) and len(built_batch) > 0 and isinstance(built_batch[0], list):
                        # batch_builder returned list of sub-batches
                        sub_batches = built_batch
                    else:
                        # Single batch
                        sub_batches = [built_batch]
                    
                    # Process each sub-batch concurrently as independent tasks
                    # Don't wait for them - immediately continue to get next batch
                    for sub_batch in sub_batches:
                        if sub_batch:
                            task = asyncio.create_task(process_sub_batch(sub_batch))
                            active_tasks.add(task)
                            
                            # Limit number of active tasks to prevent memory issues
                            # Allow up to concurrency * 2 active tasks per worker
                            max_active_tasks = max(concurrency * 2, 10)
                            if len(active_tasks) >= max_active_tasks:
                                # Wait for at least one task to complete
                                done, pending = await asyncio.wait(active_tasks, return_when=asyncio.FIRST_COMPLETED)
                                active_tasks = pending
            finally:
                # Wait for all active tasks to complete before worker exits
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
        
        # Start worker tasks
        workers = [
            asyncio.create_task(process_batch_worker())
            for _ in range(concurrency)
        ]
        
        # Wait for all workers to complete
        await asyncio.gather(*workers, return_exceptions=True)
        
        self.processing = False
        
        # Verify all items were processed
        if not self.queue.empty():
            remaining = []
            while not self.queue.empty():
                remaining.append(await self.queue.get())
            logger.warning(f"Queue not empty after processing: {len(remaining)} items remaining")
            # Put remaining items back
            for item in remaining:
                await self.queue.put(item)
        
        return results
    
    async def _requeue_item(self, item: BatchItem):
        """Requeue an item for retry, respecting max_retries."""
        if self.max_retries > 0 and item.retry_count >= self.max_retries:
            logger.error(f"Item {item.key} exceeded max retries ({self.max_retries})")
            self.stats['failed_items'] += 1
            return
        
        item.retry_count += 1
        self.stats['retry_count'] += 1
        await self.queue.put(item)
    
    async def _call_callback(self, callback: Callable, *args):
        """Safely call a callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return {
            'total_items': self.stats['total_items'],
            'processed_items': self.stats['processed_items'],
            'failed_items': self.stats['failed_items'],
            'retry_count': self.stats['retry_count'],
            'queue_size': self.queue.qsize(),
            'success_rate': (
                f"{(self.stats['processed_items'] / max(1, self.stats['total_items'])):.2%}"
                if self.stats['total_items'] > 0 else "0%"
            ),
        }


class TokenAwareBatchBuilder:
    """
    Batch builder that respects token limits.
    Useful for LLM-based translators that have token constraints.
    """
    
    def __init__(
        self,
        max_tokens: int,
        estimate_tokens: Callable[[Any], int],
        max_items: Optional[int] = None,
        base_overhead: int = 200,
        per_item_overhead: int = 20,
    ):
        """
        Initialize token-aware batch builder.
        
        Args:
            max_tokens: Maximum tokens per batch
            estimate_tokens: Function to estimate tokens for an item
            max_items: Optional maximum items per batch
            base_overhead: Base token overhead per batch
            per_item_overhead: Per-item token overhead
        """
        self.max_tokens = max_tokens
        self.estimate_tokens = estimate_tokens
        self.max_items = max_items
        self.base_overhead = base_overhead
        self.per_item_overhead = per_item_overhead
    
    def __call__(self, items: List[BatchItem]) -> List[List[BatchItem]]:
        """
        Build batches from items respecting token limits.
        Returns a list of batches (each batch is a list of BatchItems).
        """
        batches: List[List[BatchItem]] = []
        current_batch: List[BatchItem] = []
        current_tokens = self.base_overhead
        
        for item in items:
            item_tokens = self.estimate_tokens(item.value) + self.per_item_overhead
            
            # Check if adding this item would exceed limits
            if ((self.max_items and len(current_batch) >= self.max_items) or
                current_tokens + item_tokens > self.max_tokens):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [item]
                current_tokens = self.base_overhead + item_tokens
            else:
                current_batch.append(item)
                current_tokens += item_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches if batches else [items]

