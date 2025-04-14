import pandas as pd
from pydantic import BaseModel
from genlm.eval.core import Dataset


class PatternMatchingInstance(BaseModel):
    """Schema for pattern matching instance."""

    pattern: str
    pattern_id: int

    def __repr__(self):
        return f"pattern: {self.pattern} (id: {self.pattern_id})"


class PatternMatchingDataset(Dataset[PatternMatchingInstance]):
    """Dataset for pattern matching evaluation."""

    def __init__(self, patterns):
        """Initialize the dataset with a list of regex patterns.

        Args:
            patterns (list[str]): List of regex patterns to evaluate.
        """
        self.patterns = patterns

    @classmethod
    def from_csv(cls, csv_path, pattern_column):
        """Load patterns from a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            pattern_column (str): Name of the column containing regex patterns.

        Returns:
            (PatternMatchingDataset): Dataset initialized with patterns from the CSV.
        """
        patterns = pd.read_csv(csv_path)[pattern_column].to_list()
        return cls(patterns)

    def __iter__(self):
        """Iterate over regex patterns.

        Returns:
            (Iterator[PatternMatchingInstance]): Iterator over regex instances.
        """
        for pattern_id, pattern in enumerate(self.patterns):
            yield PatternMatchingInstance(pattern=pattern, pattern_id=pattern_id)

    @property
    def schema(self):
        """Get the schema class for this dataset.

        Returns:
            (type[PatternMatchingInstance]): The Pydantic model class for pattern matching instances.
        """
        return PatternMatchingInstance
