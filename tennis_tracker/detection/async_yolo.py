from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np

from ..state import Detection
from .yolo_detector import YOLODetector


class AsyncYOLOWorker:
    def __init__(self, detector: YOLODetector):
        self.detector = detector
        self._requests: Queue[np.ndarray] = Queue(maxsize=1)
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._lock = Lock()
        self._latest: Optional[Detection] = None
        self._busy = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._loop, name="yolo-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def request(self, frame: np.ndarray) -> bool:
        if self._stop.is_set():
            return False
        if self._busy or self._requests.full():
            return False
        self._requests.put(frame.copy())
        return True

    def pop_latest(self) -> Optional[Detection]:
        with self._lock:
            detection = self._latest
            self._latest = None
            return detection

    @property
    def busy(self) -> bool:
        return self._busy

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._requests.get(timeout=0.05)
            except Empty:
                continue

            self._busy = True
            try:
                try:
                    detection = self.detector.detect(frame)
                except Exception as exc:
                    print(f"[YOLO] worker stopped: {exc}")
                    self._stop.set()
                    detection = None
                if detection is not None:
                    with self._lock:
                        self._latest = detection
            finally:
                self._busy = False
                self._requests.task_done()
