import json
from uuid import UUID
from ipaddress import IPv4Address, IPv6Address
from jsonschema import Draft202012Validator, FormatChecker, SchemaError

from genlm.eval.core import Evaluator, EvaluationResult
from .dataset import JSONSchemaBenchInstance

format_checker = FormatChecker()


@format_checker.checks("ipv4")
def ipv4_check(value):
    IPv4Address(value)


@format_checker.checks("ipv6")
def ipv6_check(value):
    IPv6Address(value)


@format_checker.checks("uuid")
def uuid_check(value):
    UUID(value)


def is_json_schema_valid(schema):
    try:
        Draft202012Validator.check_schema(schema)
        return True
    except SchemaError:
        return False


def validate_json_object(instance, schema):
    if not is_json_schema_valid(schema):
        raise ValueError("Invalid JSON schema")

    validator = Draft202012Validator(schema, format_checker=format_checker)
    try:
        validator.validate(instance)

    # we catch all exceptions include ValidationError and Error from extension validators
    except Exception:
        return False
    return True


class JSONSchemaBenchEvaluator(Evaluator[JSONSchemaBenchInstance]):
    """Evaluator for JSON schema."""

    def evaluate_sample(self, instance, response):
        """Evaluate if a response is valid against the JSON schema.

        Args:
            instance (JSONSchemaBenchInstance): The JSON schema instance being evaluated.
            response (str): The model's response text.

        Returns:
            (EvaluationResult): Evaluation result for whether the response is valid against the JSON schema.
        """
        try:
            json_object = json.loads(response)
        except json.JSONDecodeError:
            return EvaluationResult(score=0, desc="invalid json")
        is_valid = validate_json_object(json_object, instance.json_schema)
        return EvaluationResult(
            score=int(is_valid),
            desc="valid" if is_valid else "json does not match schema",
        )
