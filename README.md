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

## Quickstart Example

The following standard project data layout is what all examples expect:
```text
DSEngine/
├── data/                        ← PUT USER DATASETS HERE
│   ├── customers.csv
│   ├── sales_daily.csv
│   └── large_transactions.csv
├── configs/
│   └── pipeline.yml             ← reference as: source: 'data/customers.csv'
├── outputs/                     ← REPORTS ARE WRITTEN HERE (auto-created)
│   └── full_eda/
│       └── 20240315_143022/
│           ├── report.html
│           ├── report.json
│           ├── plots/
│           │   ├── distributions_0.png
│           │   └── correlations_0.png
│           └── run.log
```

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

## Documentation
Sphinx-generated HTML documentation is included in the repository.
Open `docs/build/html/index.html` in your browser.
To rebuild documentation:
```bash
cd docs/
make html
```

## Contributing
Contributions welcome. Please submit a pull request.

## License
[MIT License](LICENSE)
