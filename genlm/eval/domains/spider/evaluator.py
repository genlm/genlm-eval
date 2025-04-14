from pathlib import Path
from functools import lru_cache
from .spider.evaluator import Evaluator as BaseSpiderEvaluator

from genlm.eval.core import Evaluator
from .dataset import SpiderInstance


class SpiderEvaluator(Evaluator[SpiderInstance]):
    """Evaluator for Spider."""

    def __init__(self, raw_spider_dir, timeout=None):
        self.raw_spider_dir = Path(raw_spider_dir)
        self.evaluator = BaseSpiderEvaluator(self.raw_spider_dir, timeout=timeout)

    @lru_cache
    def cached_eval(self, x, y, db):
        return self.evaluator.evaluate(x, y, db_name=db)

    def evaluate_response(self, instance, response):
        return self.cached_eval(response, instance.gold, instance.schema_name)
