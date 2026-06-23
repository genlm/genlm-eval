# Spider 2.0 / Spider 2.0-Lite — Dataset Synthesis

A reference for the `spider2` evaluation domain: what the dataset is, how it differs from the
original Spider, and the characteristics of its BigQuery and Snowflake backends.

> Provenance of numbers below: headline stats (632 / 547 / 21.3% / 91.2%) are from the Spider 2.0
> paper (Lei et al., ICLR 2025; arXiv:2411.07763). The per-backend instance counts for
> Spider 2.0-Lite were computed directly from the official `spider2-lite.jsonl`
> (547 rows) — see [Backend breakdown](#backend-breakdown). Full-Spider-2.0 per-system *database*
> counts are as reported by the project and are noted as such.

---

## 1. What Spider 2.0 is

**Spider 2.0** (Lei et al., ICLR 2025 Oral) is a text-to-SQL benchmark built from **real-world
enterprise data workflows** rather than hand-authored academic questions. It contains **632**
problems whose databases come from production applications — often **>1,000 columns** — hosted both
locally and in cloud warehouses (BigQuery, Snowflake). Solving an instance frequently requires
reading database metadata, dialect documentation, and even project-level code; the resulting SQL
queries **frequently exceed 100 lines**.

**Spider 2.0-Lite** is the self-contained, **single-SQL-generation** variant used by this domain:
**547** instances, each shipping the database metadata and supporting documentation needed to write
one SQL query. It is the practical target for constrained-generation evaluation.

**Difficulty signal.** On Spider 2.0, `o1-preview` reaches **21.3%** execution accuracy, versus
**91.2%** on Spider 1.0 and **73.0%** on BIRD — i.e. Spider 2.0 is roughly 4× harder than the
benchmark it succeeds.

---

## 2. How Spider 2.0 differs from Spider 1.0

| Dimension | **Spider 1.0** (Yu et al., 2018) | **Spider 2.0** (Lei et al., 2025) |
|---|---|---|
| Task framing | One NL question → one SQL query | Real enterprise **workflows**; often multi-step, doc/codebase-grounded |
| Size | ~10,181 questions / 200 DBs / 138 domains | 632 problems (full); **Spider 2.0-Lite = 547** |
| Backends | **SQLite only** | **BigQuery + Snowflake + SQLite** (Lite); + DuckDB/Postgres/ClickHouse in full |
| Schema scale | Small — a few tables, tens of columns | **Often >1,000 columns**; nested/repeated structures |
| SQL complexity | Mostly short; few joins | Frequently **>100 lines**; deep nesting, dialect-specific functions |
| External knowledge | None — schema is self-contained | **Required** — metadata, dialect docs, supplied knowledge files |
| SQL dialect | Single (SQLite) | **Multiple** (GoogleSQL / Snowflake SQL / SQLite) |
| Top-LLM accuracy | ~91% | ~21% |

**Why the gap matters for this repo.** Three of these differences directly shape the
constrained-generation work:
1. **Multiple dialects** — a single SQLite grammar/potential no longer covers the benchmark.
2. **Schema scale** — dumping a full 1,000-column schema into the prompt is infeasible; schema
   linking becomes necessary.
3. **External knowledge** — the question alone is underspecified without the supplied docs.

---

## 3. Spider 2.0-Lite layout & backend routing

```
spider2-lite/
  spider2-lite.jsonl                     # one instance per line: {instance_id, db, question, external_knowledge}
  evaluation_suite/
    gold/sql/{instance_id}.sql           # gold query
    gold/exec_result/{instance_id}*.csv  # gold result table(s) — the comparison target
    spider2lite_eval.jsonl               # per-instance {condition_cols, ignore_order}
    snowflake_credential.json            # (you supply)
  resource/
    databases/{backend}/{db}/DDL.csv     # schema as CREATE TABLE statements, per backend
    databases/spider2-localdb/{db}.sqlite# downloadable SQLite files (local slice only)
    documents/                           # external-knowledge files referenced per instance
```

**There is no `db_type` field.** The backend is encoded in the **`instance_id` prefix** (the
convention the official suite and this domain's `backend_for_instance()` both use):

| Prefix | Backend | Executes on |
|---|---|---|
| `local*` | SQLite | your machine (downloaded `.sqlite` file) |
| `sf_bq*`, `sf*` | Snowflake | Snowflake cloud (host-provided warehouse) |
| `bq*`, `ga*` | BigQuery | Google Cloud (`bigquery-public-data`) |

**Scoring** compares the predicted query's *execution result* against the shipped gold
`exec_result` CSV(s) (numeric tolerance, optional row-order-insensitivity, optional
`condition_cols` subset) — gold SQL need not be re-executed.

### Backend breakdown

Computed from the official 547-row `spider2-lite.jsonl`:

| Backend | Instances | Fraction | Raw prefixes |
|---|---:|---:|---|
| **Snowflake** | 207 | **37.8%** | `sf_bq` (189) + `sf` (18) |
| **BigQuery** | 205 | **37.5%** | `bq` (180) + `ga` (25) |
| **SQLite (local)** | 135 | **24.7%** | `local` (135) |
| **Total** | 547 | 100% | |

It is an almost even three-way split. The local SQLite instances are **real** Spider 2.0 data
(not just samples). This domain enables `{sqlite, snowflake}` by default — **62.5%** of the
benchmark with no GCP billing — and treats BigQuery as opt-in.

---

## 4. BigQuery backend characteristics

- **What it is.** Google's serverless, columnar cloud data warehouse. Spider 2.0's BigQuery
  instances query the public **`bigquery-public-data`** project — e.g. Google Analytics
  (`ga4`, `ga360`), `github_repos`, patents, NOAA weather, blockchain/crypto. Datasets are
  **terabyte-to-petabyte scale** and **cannot be downloaded** — you query them in place.
- **Dialect: GoogleSQL.** Standard-SQL core plus BigQuery-specific features — `STRUCT`/`ARRAY`
  nested & repeated fields, `UNNEST(...)`, backtick-quoted `` `project.dataset.table` `` names,
  and BQ date/analytic functions. GA datasets in particular lean heavily on nested records.
- **Access — free via Sandbox.** Sign in at the BigQuery console, create/select a project
  *without* attaching billing → **Sandbox mode** (no credit card). Star `bigquery-public-data`,
  then create a **service account** and download its **JSON key** (`{service_account}-xxxx.json`)
  for programmatic access (`google-cloud-bigquery`).
- **Cost model.** Billed by **bytes *scanned*** (not rows returned); the standard free tier is
  **1 TiB query processing / month**, and public-dataset storage is free. **Caveat:** a single
  poorly-pruned query over a multi-TB table can consume a large share of that allowance; Sandbox
  caps usage (queries fail rather than bill) — exceeding it requires enabling billing
  (~$6.25 / TiB scanned thereafter).
- **Coverage.** The public project covers ~70% of the benchmark's BigQuery SQLs.

## 5. Snowflake backend characteristics

- **What it is.** A cloud data warehouse with **host-provided compute**. Spider 2.0's `sf_bq*`
  instances are **Snowflake mirrors of the BigQuery public datasets** (hence the `sf_bq` name);
  `sf*` are Snowflake-native.
- **Dialect: Snowflake SQL.** ANSI-like, with Snowflake-specific features — the `VARIANT`
  semi-structured type, `LATERAL FLATTEN(...)` for nested data, `QUALIFY`, and Snowflake function
  names. Tables are referenced **fully qualified** as `DATABASE.SCHEMA.TABLE`, so no `USE DATABASE`
  is needed before running a query.
- **Access — free, host-managed.** Submit the Spider 2 *"Snowflake Access"* Google Form with a
  desired username + email; credentials arrive in **~12 h**. Set up MFA, reset the password, then
  generate a **programmatic token** (365-day expiry). The Spider 2 hosts provide the compute
  warehouse, so **there is no billing on your side**.
- **Credential file** (`snowflake_credential.json`): `username`, `password` (the token),
  `account: "RSRSBDK-YDB67606"`, `role: "PARTICIPANT"`, `warehouse: "COMPUTE_WH_PARTICIPANT"`.

### BigQuery vs Snowflake at a glance

| | **BigQuery** | **Snowflake** |
|---|---|---|
| Provider | Google Cloud | Snowflake (Spider 2-hosted) |
| Who owns the account | **You** (Sandbox/free or billed) | **Spider 2 hosts** (participant role) |
| Who pays compute | You (free tier 1 TiB/mo, then per-byte) | The hosts (free to you) |
| Access friction | Console + service-account key (immediate) | Google Form, ~12 h wait |
| Dialect | GoogleSQL (`UNNEST`, `STRUCT`/`ARRAY`) | Snowflake SQL (`FLATTEN`, `VARIANT`, `QUALIFY`) |
| Table refs | `` `project.dataset.table` `` | `DATABASE.SCHEMA.TABLE` (fully qualified) |
| Lite share | 37.5% (205) | 37.8% (207) |

---

## 6. Status in this domain (`genlm/eval/domains/spider2/`)

- **SQLite** — executes locally; no credentials.
- **Snowflake** — implemented (`execute_snowflake`), dispatched by `instance_id` prefix; needs
  `snowflake_credential.json`. Live run pending Snowflake access.
- **BigQuery** — implemented (`execute_bigquery`); **opt-in** (executes only when a
  `bigquery_project` is set) and authenticates via Application Default Credentials. Live execution
  against `bigquery-public-data` has been verified end-to-end.
- **Potential** — currently the Spider 1 SQLite lark grammar (`Spider2TableColumnVerifier`),
  adequate for the local slice only. A dialect-aware potential + schema linking for large schemas
  is future work.

---

## Sources

- Spider 2.0 paper — Lei et al., *Spider 2.0: Evaluating Language Models on Real-World Enterprise
  Text-to-SQL Workflows*, ICLR 2025. arXiv:2411.07763.
- Project site — https://spider2-sql.github.io/
- Repository & guidelines — https://github.com/xlang-ai/Spider2
  (`spider2-lite/`, `assets/Snowflake_Guideline.md`, `assets/Bigquery_Guideline.md`).
- Per-backend counts computed from `spider2-lite/spider2-lite.jsonl` (547 rows).
