# DSEngine

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![PyPI Version](https://img.shields.io/badge/pypi-v1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

DSEngine is a declarative data science library. Users define what they want in a YAML config file and run one command — the library handles everything. It is designed as a companion to MLEngine, handling data loading, exploration, and statistical validation in a fully automated, transparent pipeline.

## Library Structure
| Sub-package | Responsibility |
|---|---|
| `data` | Loading, pre-flight data inspection, validation, and sampling. |
| `exploration` | Comprehensive EDA: summaries, missing values, distributions, correlations, and outliers. |
| `statistics` | Statistical hypothesis testing, normality checks, and relationship significance. |
| `time_series` | Trend decomposition, stationarity testing, and feature generation for time series. |
| `reporting` | Output generation, saving plots, and building HTML/JSON reports. |
| `utils` | Core engine utilities: configuration parsing, pipeline runner, logging, and unified plot styling. |

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
│   ├── data/                   ← Data ingestion and preparation
│   ├── exploration/            ← EDA functional blocks
│   ├── statistics/             ← Statistical analysis blocks
│   ├── time_series/            ← Time series analysis blocks
│   ├── reporting/              ← Output and report generation
│   └── utils/                  ← Internal infrastructure
├── configs/                    ← User-facing configuration files
│   ├── pipeline.yml            ← User defines experiments here
│   ├── steps_defaults.yml      ← Default parameters for each step type
│   └── schema_template.yml     ← Template for data validation schemas
├── examples/                   ← Example notebooks and pipeline runner
│   ├── run_pipeline.py         ← CLI entry point
│   ├── datasets/               ← PUT USER DATASETS HERE
│   ├── 00_Download_Datasets.ipynb
│   ├── 01_Basic_EDA_Declarative.ipynb
│   ├── 02_Time_Series_Analysis_Declarative.ipynb
│   ├── 03_Statistical_Testing_Declarative.ipynb
│   ├── 04_Full_Pipeline_Declarative.ipynb
│   └── 05_Advanced_Data_Science_Declarative.ipynb
├── docs/                       ← Sphinx documentation
├── requirements.txt
├── setup.py
├── LICENSE                     ← MIT License text
└── README.md
```

## Quickstart Example

Create a `pipeline.yml` in the `configs/` directory:
```yaml
my_first_experiment:
  data:
    source: 'data/customers.csv'
  steps:
    - name: 'overview'
      type: 'summary'
      columns: []
      params: {}
    - name: 'missing'
      type: 'missing'
      columns: []
      params:
        plot_type: 'both'
    - name: 'distributions'
      type: 'distributions'
      columns: ['age', 'income']
      params:
        plot_type: 'both'
  output:
    path: 'outputs/'
    format: ['html', 'json']
    save_plots: true
```

Run the pipeline:
```bash
python examples/run_pipeline.py --experiment my_first_experiment
```

## Interactive Notebooks

DSEngine ships with pre-configured Jupyter notebooks in the `examples/` directory that demonstrate each core module. Rather than rewriting code, these interactive notebooks load the YAML files inside the `configs/` folder and automatically run the entire declarative pipeline.

To use them:
1. Place your dataset in `examples/datasets/`.
2. Edit the corresponding YAML file in `configs/` (e.g. `configs/01_basic_eda.yml`) to point `data.source` to your dataset.
3. Open the corresponding example notebook (e.g. `00_Basic_EDA.ipynb`) and run the cells.
4. The notebook will automatically compile the pipeline run and embed the generated analytical HTML reports directly inside of your Jupyter environment!

## Documentation
Sphinx-generated HTML documentation is included in the repository.
Open `docs/build/html/index.html` in your browser.
To rebuild documentation:
```bash
cd docs/
python -m sphinx -b html source build/html
```

## Contributing
Contributions welcome. Please submit a pull request.

## License
[MIT License](LICENSE)
