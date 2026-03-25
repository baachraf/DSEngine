# DSEngine

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![PyPI Version](https://img.shields.io/badge/pypi-v1.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DSEngine is a declarative data science library. Users define what they want in a YAML config file and run one command — the library handles everything. It is designed as a companion to MLEngine, handling data loading, preparation, exploration, and statistical validation in a fully automated, transparent pipeline.

## Library Structure
| Sub-package | Responsibility |
|---|---|
| `data` | Loading (Files & SQL), pre-flight inspection, validation, and sampling. |
| `preparation` | **(New)** Advanced data transformations: imputation, scaling, encoding, clipping, and power transforms. |
| `exploration` | Comprehensive EDA: summaries, missing values, distributions, correlations, and outliers. |
| `statistics` | Hypothesis testing with Effect Sizes (Cohen's d/Eta²), Post-hoc tests (Tukey), and Relationship significance. |
| `time_series` | Trend decomposition, stationarity testing, and feature generation for time series. |
| `reporting` | Output generation, saving plots, and building HTML/JSON reports. |
| `utils` | Core engine utilities: configuration parsing, pipeline runner, and unified plot styling. |

## Installation
```bash
# Clone the repository and install dependencies
git clone https://github.com/your-org/DSEngine.git
cd DSEngine
pip install -r requirements.txt
pip install -e .
```

## Repository Structure

The DSEngine project strictly follows this blueprint layout:
```text
DSEngine/
├── ds_engine/                  ← Core library package
│   ├── data/                   ← Data ingestion (CSV, Excel, JSON, Parquet, SQL)
│   ├── preparation/            ← Data cleaning and transformation blocks
│   ├── exploration/            ← EDA functional blocks
│   ├── statistics/             ← Statistical analysis and hypothesis testing
│   ├── time_series/            ← Time series analysis blocks
│   ├── reporting/              ← Output and report generation
│   └── utils/                  ← Internal infrastructure
├── configs/                    ← User-facing configuration files
│   ├── pipeline.yml            ← User defines experiments here
│   ├── steps_defaults.yml      ← Default parameters for each step type
│   └── ...
├── examples/                   ← Example notebooks and pipeline runner
│   ├── run_pipeline.py         ← CLI entry point
│   ├── datasets/               ← Sample datasets (CSV & SQLite)
│   ├── 00_Download_Datasets.ipynb
│   ├── 01_Basic_EDA_Declarative.ipynb
│   ├── 02_Time_Series_Analysis_Declarative.ipynb
│   ├── 03_Statistical_Testing_Declarative.ipynb
│   ├── 04_Full_Pipeline_Declarative.ipynb
│   ├── 05_Advanced_Data_Science_Declarative.ipynb
│   ├── 06_Data_Preparation_Declarative.ipynb
│   ├── 07_Advanced_Transforms_Declarative.ipynb
│   ├── 08_Hyperparams_Declarative.ipynb
│   └── 09_Database_Loading_Declarative.ipynb
├── requirements.txt            ← Includes SQLAlchemy for DB support
├── setup.py
├── LICENSE                     ← MIT License text
└── README.md
```

## Core Features

### 🛠️ Declarative "Zero-Code" Architecture
DSEngine eliminates the need for boilerplate Python code. By separating the **experiment logic** (YAML) from the **execution engine** (Python), data scientists can iterate faster and maintain perfect reproducibility. 
- **Universal Contract**: Every functional block follows a strict `(data, plots, metrics)` contract.
- **Dynamic Registry**: New preparation or analysis steps can be registered and used instantly via YAML.
- **Hyperparameter Testing**: Define multiple experiments in a single file to compare strategies side-by-side.

### 🧪 Advanced Data Preparation & Engineering
The `ds_engine.preparation` module treats data cleaning as a first-class citizen of the pipeline.
- **Statistical Imputation**: Intelligent handling of missing data using `mean`, `median`, `mode`, or `constant` strategies.
- **Feature Scaling**: Industry-standard normalization including `StandardScaler`, `MinMaxScaler`, and `RobustScaler` (outlier-resistant).
- **Categorical Engineering**: Automated `One-Hot` and `Label` encoding to bridge the gap between raw data and ML-ready tensors.
- **Non-Linear Transforms**: Advanced Yeo-Johnson and Box-Cox power transforms, plus standard Log, Log1p, and Sqrt operations.
- **Outlier Orchestration**: Configurable IQR and Z-score clipping to stabilize distributions with zero manual effort.

### 🗄️ Enterprise SQL Database Ingestion
Native integration with **SQLAlchemy** allows DSEngine to sit directly on top of your enterprise data stack. 
- **Multi-Dialect Support**: Seamlessly connect to PostgreSQL, MySQL, SQLite, Oracle, and MSSQL.
- **Transparent Loading**: Simply provide a `sqlite:///`, `postgresql://`, or similar URI in `data.source`.
- **Flexible Schema Fetching**: Pull entire tables (`table: "users"`) or execute complex queries (`query: "SELECT ... JOIN ..."`) via `loader_params`.
- **Pre-flight Validation**: Automatic inspection of schema alignment and data types before the pipeline begins.

### 📊 Expert Statistical Validation
DSEngine doesn't just run tests; it provides the context needed for scientific decision-making.
- **Beyond p-values**: Automatic calculation of **Effect Sizes** (Cohen's d for T-tests, Eta-squared for ANOVA) to measure the magnitude of findings.
- **Automated Post-Hocs**: When ANOVA reveals significance, the engine automatically executes **Tukey HSD** to identify specific group differences.
- **Multivariate Relationships**: Deep correlation analysis using Pearson, Spearman, and **Cramer's V** (for categorical associations).
- **Assumptions Checking**: Integrated Shapiro-Wilk and Levene's tests verify normality and homoscedasticity before recommending specific methodologies.

### 📱 Interactive Visual Intelligence
Generate professional-grade analytical assets automatically.
- **Unified Styling**: All plots follow a curated, premium design system with high-contrast color palettes and modern typography.
- **HTML + JSON Reports**: Comprehensive summaries including interactive data tables, high-resolution figures, and raw metric exports for downstream consumption.
- **Notebook Embedding**: Direct integration allows HTML reports to render natively inside Jupyter cells for an interrupted flow.

## Quickstart Example

Run the pipeline via CLI:
```bash
python examples/run_pipeline.py --config configs/my_config.yml --experiment my_experiment
```

## Interactive Notebooks

DSEngine ships with pre-configured Jupyter notebooks in the `examples/` directory. Rather than rewriting code, these notebooks load the YAML files and automatically embedded the generated reports directly inside your Jupyter environment!

## Documentation
Sphinx-generated HTML documentation is included in the repository. Open `docs/build/html/index.html` in your browser.

## License
[MIT License](LICENSE)
