import os
import json
from datetime import datetime
from src.logger import get_logger
logger = get_logger(__name__)
class RateLimiter:
    def __init__(self, limit: int = 900):
        self.limit = limit
        self.storage_file = "data/api_usage.json"
        os.makedirs('data', exist_ok=True)
        self._load()
    def _load(self):
        today = datetime.now().strftime('%Y-%m-%d')
        if os.path.exists(self.storage_file):
            with open(self.storage_file) as f:
                data = json.load(f)
            if data.get('date') == today:
                self.calls_today = data.get('calls', 0)
                return
        self.calls_today = 0
    def _save(self):
        today = datetime.now().strftime('%Y-%m-%d')
        with open(self.storage_file, 'w') as f:
            json.dump({'date': today, 'calls': self.calls_today}, f)
    def can_call(self) -> bool:
        return self.calls_today < self.limit
    def record_call(self):
        self.calls_today += 1
        self._save()
        logger.info(f"API call recorded. {self.calls_today}/{self.limit} used today.")
    @property
    def remaining(self) -> int:
        return self.limit - self.calls_today