![Logo](assets/logo.png)

<p align="center">
<a href="https://genlm.github.io/genlm-eval/"><img alt="Tests" src="https://github.com/genlm/genlm-eval/actions/workflows/pytest.yml/badge.svg"/></a>
<a href="https://codecov.io/github/genlm/genlm-eval"><img alt="codecov" src="https://codecov.io/github/genlm/genlm-eval/graph/badge.svg?token=665ffkDXvZ"/></a>
</p>

A flexible framework for evaluating constrained generation models, built for the GenLM ecosystem. This library provides standardized interfaces and benchmarks for assessing model performance across various constrained generation tasks.

## Features

- **Standardized Interfaces**: Define and evaluate constrained generation tasks using a general and consistent API.
- **Benchmarking**: Compare model performance across multiple domains and metrics.
- **Customizable**: Create custom evaluation domains and models.

## Documentation

- **Getting Started**: Visit our [documentation](https://genlm.github.io/genlm-eval/) for installation and usage guides.
- **API Reference**: Browse the [API documentation](https://genlm.github.io/genlm-eval/reference/) for detailed information about the library's components.
- **Cookbook**: Check out our [examples and tutorials](https://genlm.github.io/genlm-eval/cookbook/) for:
    * Using built-in domains (Pattern Matching, Text-to-SQL, Molecular Synthesis)
    * Creating custom evaluation domains

## Components

- **Datasets**: Specifies and iterates over the dataset instances of a constrained generation task.
- **Evaluators**: Evaluates the model's output.
- **Model Adapters**: Wraps the model to provide a unified interface for evaluation.
- **Runners**: Orchestrates the evaluation process with output caching.

## Installation

```bash
git clone https://github.com/genlm/genlm-eval.git
cd genlm-eval
pip install -e .
```

For domain-specific dependencies, refer to the [examples and tutorials](https://genlm.github.io/genlm-eval/cookbook/).
