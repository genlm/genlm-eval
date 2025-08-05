import json
from datasets import load_dataset
from genlm.eval.core import Dataset, Instance
from typing import Union


###########
# Dataset #
###########


class JSONSchemaBenchInstance(Instance):
    """Schema for JSON instance."""

    json_schema: dict
    instance_id: Union[int, str]
    task: str

    def __repr__(self):
        return f"schema: {self.json_schema} (id: {self.instance_id})"


class JSONSchemaBenchDataset(Dataset[JSONSchemaBenchInstance]):
    """Dataset for JSON evaluation using the JSONSchemaBench dataset."""

    def __init__(self, schemas, tasks, unique_ids):
        """Initialize the dataset with a list of regex patterns.

        Args:
            schemas (list[str]): The JSON schemas to evaluate.
            tasks (list[str]): The task name for each schema.
            unique_ids (list[int]): The unique id for each schema.
        """
        assert len(schemas) == len(tasks)
        self.schemas = schemas
        self.tasks = tasks
        self.unique_ids = unique_ids

    def __len__(self):
        return len(self.schemas)

    @classmethod
    def from_tasks(cls, tasks, split="val"):
        """Load tasks from a list of tasks in the JSONSchemaBench dataset.

        Args:
            tasks (list[Task]): The tasks to evaluate.
            split (str): The split to load. Defaults to "val".

        Returns:
            (JSONSchemaBenchDataset): Dataset initialized with schemas from the tasks.
        """
        all_schemas = []
        task_names = []
        all_unique_ids = []

        for task in tasks:
            dataset = load_dataset("epfl-dlab/JSONSchemaBench", task)[split]
            schemas = dataset["json_schema"]
            unique_ids = dataset["unique_id"]
            assert len(schemas) == len(unique_ids)
            for schema, unique_id in zip(schemas, unique_ids):
                assert unique_id not in all_unique_ids
                all_unique_ids.append(unique_id)
                all_schemas.append(json.loads(schema))
                task_names.append(task)

        return cls(all_schemas, task_names, all_unique_ids)

    def __iter__(self):
        """Iterate over JSON schemas.

        Returns:
            (Iterator[JSONSchemaBenchInstance]): Iterator over JSON schema instances.
        """
        for schema_id, schema in enumerate(self.schemas):
            yield JSONSchemaBenchInstance(
                json_schema=schema,
                instance_id=self.unique_ids[schema_id],
                task=self.tasks[schema_id],
            )

    @property
    def schema(self):
        """Get the schema class for this dataset.

        Returns:
            (type[JSONSchemaBenchInstance]): The Pydantic model class for JSON schema instances.
        """
        return JSONSchemaBenchInstance
