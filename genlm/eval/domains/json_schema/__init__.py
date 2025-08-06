import json

from .dataset import JSONSchemaBenchDataset, JSONSchemaBenchInstance
from .evaluator import JSONSchemaBenchEvaluator


def few_shots_messages_formatter(task: str, schema: dict, system_prompt: str):
    examples = [value for key, value in EXAMPLES_FOR_TASK.items() if task in key]
    messages = [{"role": "system", "content": system_prompt}]
    for task_examples in examples:
        for input, output in task_examples:
            messages.append({"role": "user", "content": input})
            messages.append({"role": "assistant", "content": output})
    messages.append({"role": "user", "content": json.dumps(schema)})
    return messages


DEFAULT_SYSTEM_PROMPT = "You need to generate a JSON object that matches the schema below. Output the JSON object on a single line. DO NOT use multiple lines and DO NOT output any other text."

# Examples taken from JsonSchemaBench dataset, but stripped of the \n characters.

EXAMPLES_FOR_TASK = {
    ("Snowplow",): [
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a JSON Paths file for loading Redshift from JSON or Avro, http://docs.aws.amazon.com/redshift/latest/dg/copy-parameters-data-format.html#copy-json-jsonpaths",\n    "properties": {\n        "jsonpaths": {\n            "items": {\n                "type": "string"\n            },\n            "minItems": 1,\n            "type": "array"\n        }\n    },\n    "required": [\n        "jsonpaths"\n    ],\n    "self": {\n        "format": "jsonschema",\n        "name": "jsonpaths_file",\n        "vendor": "com.amazon.aws.redshift",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"jsonpaths": ["$.user.id", "$.user.name", "$.user.address.street"]}',
        ),
        (
            '{\n    "additionalProperties": false,\n    "description": "Schema for a Google Analytics enhanced e-commerce product impression custom metric entity",\n    "properties": {\n        "customMetricIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "listIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "productIndex": {\n            "maximum": 200,\n            "minimum": 1,\n            "type": "integer"\n        },\n        "value": {\n            "type": [\n                "integer",\n                "null"\n            ]\n        }\n    },\n    "self": {\n        "format": "jsonschema",\n        "name": "product_impression_custom_metric",\n        "vendor": "com.google.analytics.measurement-protocol",\n        "version": "1-0-0"\n    },\n    "type": "object"\n}',
            '{"customMetricIndex": 120, "listIndex": 45, "productIndex": 10, "value": 300}',
        ),
    ],
    ("Github_easy", "Github_hard", "Github_medium", "Github_trivial", "Github_ultra"): [
        (
            '{\n    "$schema": "http://json-schema.org/draft-04/schema#",\n    "definitions": {\n        "address1": {"type": "string"},\n        "address2": {"type": "string"},\n        "city": {"type": "string"},\n        "country": {"type": "string"},\n        "postalCode": {"type": "string"},\n        "state": {"type": "string"}\n    },\n    "description": "A simple address schema",\n    "properties": {\n        "address1": {"$ref": "#/definitions/address1"},\n        "address2": {"$ref": "#/definitions/address2"},\n        "city": {"$ref": "#/definitions/city"},\n        "country": {"$ref": "#/definitions/country"},\n        "postalCode": {"$ref": "#/definitions/postalCode"},\n        "state": {"$ref": "#/definitions/state"}\n    },\n    "type": "object"\n}',
            '{"address1": "123 Main Street", "address2": "Apt 4B", "city": "Seattle", "country": "USA", "postalCode": "98101", "state": "WA"}',
        ),
        (
            '{\n    "$schema": "http://json-schema.org/draft-06/schema#",\n    "definitions": {\n        "ElementType": {\n            "enum": ["component", "directive"],\n            "type": "string"\n        },\n        "SelectorChange": {\n            "properties": {\n                "remove": {\n                    "description": "Remove directive/component",\n                    "type": "boolean"\n                },\n                "replaceWith": {\n                    "description": "Replace original selector with new one",\n                    "type": "string"\n                },\n                "selector": {\n                    "description": "Original selector to apply change to",\n                    "type": "string"\n                },\n                "type": {\n                    "$ref": "#/definitions/ElementType",\n                    "description": "Type of selector the change applies to - either component or directive"\n                }\n            },\n            "required": ["selector", "type"],\n            "type": "object"\n        }\n    },\n    "properties": {\n        "changes": {\n            "description": "An array of changes to component/directive selectors",\n            "items": {\n                "$ref": "#/definitions/SelectorChange"\n            },\n            "type": "array"\n        }\n    },\n    "required": ["changes"],\n    "type": "object"\n}',
            '{"changes": [{"selector": "app-root", "type": "component", "remove": false, "replaceWith": "new-root"}, {"selector": "my-directive", "type": "directive", "remove": true, "replaceWith": "new-directive"}]}',
        ),
    ],
    ("Glaiveai2K",): [
        (
            '{"properties": {"username": {"description": "The user\'s username", "type": "string"}, "email": {"description": "The user\'s email address", "type": "string"}, "age": {"description": "The user\'s age", "type": "integer"}, "is_active": {"description": "Whether the user is active", "type": "boolean"}}, "required": ["username", "email"], "type": "object"}',
            '{"username": "johndoe", "email": "john@example.com", "age": 30, "is_active": true} ',
        ),
        (
            '{"properties": {"product_id": {"description": "The ID of the product", "type": "string"}, "rating": {"description": "The rating given by the user", "type": "integer"}, "comments": {"description": "Additional comments about the product", "type": "string"}}, "required": ["product_id", "rating"], "type": "object"}',
            '{"product_id": "12345", "rating": 5, "comments": "Excellent product! Highly recommend."} ',
        ),
    ],
    ("JsonSchemaStore",): [
        (
            '{\n  "$id": "https://json.schemastore.org/minecraft-trim-pattern.json",\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "description": "A trim pattern for a Minecraft data pack config schema",\n  "properties": {\n    "asset_id": {\n      "type": "string"\n    },\n    "description": {\n      "properties": {\n        "color": {\n          "type": "string"\n        },\n        "translate": {\n          "type": "string"\n        }\n      },\n      "required": ["translate"],\n      "type": "object"\n    },\n    "template_item": {\n      "type": "string"\n    }\n  },\n  "required": ["asset_id", "description", "template_item"],\n  "title": "Minecraft Data Pack Trim Pattern",\n  "type": "object"\n}',
            '{"asset_id": "minecraft:trim_pattern", "description": {"color": "#FFAA00", "translate": "trim_pattern.description"}, "template_item": "minecraft:template_item"}',
        ),
        (
            '{\n  "$comment": "https://minecraft.fandom.com/wiki/Data_Pack",\n  "$id": "https://json.schemastore.org/minecraft-damage-type.json",\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "description": "A damage type\'s for a Minecraft data pack config schema",\n  "properties": {\n    "death_message_type": {\n      "enum": ["default", "fall_variants", "intentional_game_design"],\n      "type": "string"\n    },\n    "effects": {\n      "enum": ["hurt", "thorns", "drowning", "burning", "poking", "freezing"],\n      "type": "string"\n    },\n    "exhaustion": {\n      "type": "number"\n    },\n    "message_id": {\n      "type": "string"\n    },\n    "scaling": {\n      "enum": ["never", "always", "when_caused_by_living_non_player"],\n      "type": "string"\n    }\n  },\n  "required": ["message_id", "scaling", "exhaustion"],\n  "title": "Minecraft Data Pack Damage Type",\n  "type": "object"\n}',
            '{"message_id": "minecraft:damage.message", "scaling": "always", "exhaustion": 0.3, "death_message_type": "default", "effects": "hurt"}',
        ),
    ],
    ("Kubernetes",): [
        (
            '{\n  "description": "A topology selector requirement is a selector that matches given label. This is an alpha feature and may change in the future.",\n  "properties": {\n    "key": {\n      "description": "The label key that the selector applies to.",\n      "type": ["string", "null"]\n    },\n    "values": {\n      "description": "An array of string values. One value must match the label to be selected. Each entry in Values is ORed.",\n      "items": {\n        "type": ["string", "null"]\n      },\n      "type": ["array", "null"]\n    }\n  },\n  "required": ["key", "values"],\n  "type": "object"\n}',
            '{"key": "region", "values": ["us-west-1", "us-east-1"]}',
        ),
        (
            '{\n  "description": "HostAlias holds the mapping between IP and hostnames that will be injected as an entry in the pod\'s hosts file.",\n  "properties": {\n    "hostnames": {\n      "description": "Hostnames for the above IP address.",\n      "items": {\n        "type": ["string", "null"]\n      },\n      "type": ["array", "null"]\n    },\n    "ip": {\n      "description": "IP address of the host file entry.",\n      "type": ["string", "null"]\n    }\n  },\n  "type": "object"\n}',
            '{"ip": "192.168.1.1", "hostnames": ["example.com", "test.com"]}',
        ),
    ],
    ("WashingtonPost",): [
        (
            '{\n  "additionalProperties": false,\n  "description": "Models a auxiliary used in targeting a piece of content.",\n  "properties": {\n    "_id": {\n      "description": "The unique identifier for this auxiliary.",\n      "type": "string"\n    },\n    "name": {\n      "description": "The general name for this auxiliary.",\n      "type": "string"\n    },\n    "uid": {\n      "description": "A short identifier for this auxiliary. Usually used in cases where a long form id cannot work.",\n      "type": "string"\n    }\n  },\n  "required": ["_id", "uid"],\n  "title": "Auxiliary",\n  "type": "object"\n}',
            '{"_id": "12345", "uid": "aux123", "name": "Sample Auxiliary"}',
        ),
        (
            '{\n  "additionalProperties": {},\n  "definitions": {\n    "trait_additional_properties_json": {\n      "$schema": "http://json-schema.org/draft-04/schema#",\n      "additionalProperties": {},\n      "description": "A grab-bag object for non-validatable data.",\n      "title": "Has additional properties",\n      "type": "object"\n    }\n  },\n  "description": "Comment configuration data",\n  "properties": {\n    "additional_properties": {\n      "$ref": "#/definitions/trait_additional_properties_json"\n    },\n    "allow_comments": {\n      "description": "If false, commenting is disabled on this content.",\n      "type": "boolean"\n    },\n    "comments_period": {\n      "description": "How long (in days) after publish date until comments are closed.",\n      "type": "integer"\n    },\n    "display_comments": {\n      "description": "If false, do not render comments on this content.",\n      "type": "boolean"\n    },\n    "moderation_required": {\n      "description": "If true, comments must be moderator-approved before being displayed.",\n      "type": "boolean"\n    }\n  },\n  "title": "Comments",\n  "type": "object"\n}',
            '{"allow_comments": true, "comments_period": 30, "display_comments": true, "moderation_required": false, "additional_properties": {}}',
        ),
    ],
    ("default",): [],
}


def default_prompt_formatter(
    tokenizer,
    instance,
    use_chat_format=True,
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    """Default prompt formatter for JSON Schema.

    Args:
        tokenizer (Tokenizer): The tokenizer to use.
        instance (JSONSchemaInstance): The instance to format.
        use_chat_format (bool): Whether to use chat format.
        system_prompt (str): The system prompt to use.

    Returns:
        (list[int]): The prompt ids.
    """
    if use_chat_format:
        return tokenizer.apply_chat_template(
            conversation=few_shots_messages_formatter(
                task=instance.task,
                schema=instance.json_schema,
                system_prompt=system_prompt,
            ),
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        raise NotImplementedError("JSON schema does not support non-chat format")


__all__ = [
    "JSONSchemaBenchDataset",
    "JSONSchemaBenchInstance",
    "JSONSchemaBenchEvaluator",
    "EXAMPLES_FOR_TASK",
    "DEFAULT_SYSTEM_PROMPT",
    "default_prompt_formatter",
]
