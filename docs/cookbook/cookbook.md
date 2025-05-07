# Cookbook

This cookbook provides examples of how to use `genlm-eval` for evaluating constrained language models on various domains. Each example demonstrates how to:

1. Set up the required dependencies
2. Initialize the dataset and evaluator
3. Define a model adaptor
4. Run the evaluation

## Available Examples

### Custom Domains
Learn how to extend `genlm-eval` to evaluate models on your own custom domains. This example walks through:
- Defining a dataset schema
- Implementing an evaluator
- Creating a model adaptor
- Running the evaluation pipeline

[View Example](custom_domains.ipynb)

### Domain-Specific Examples

#### Pattern Matching
Evaluate models on generating strings that match complex pattern specifications:
- Task: Generate strings conforming to expressive pattern-matching specifications
- Data: 400+ pattern-matching specifications with features beyond regular expressions

[View Example](domains/pattern_matching.ipynb)

#### Molecular Synthesis
Evaluate models on generating valid molecular structures:
- Task: Generate drug-like compounds using SMILES notation
- Data: Few-shot prompts from GDB-17 database

[View Example](domains/molecular_synthesis.ipynb)

#### Text to SQL (Spider)
Evaluate models on generating SQL queries from natural language:
- Task: Generate SQL queries from natural language questions
- Data: Spider dataset with database schemas

[View Example](domains/spider.ipynb)
