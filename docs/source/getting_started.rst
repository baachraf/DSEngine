Getting Started
===============

Installation
------------
Clone the repository and install:

.. code-block:: bash

    git clone https://github.com/your-org/DSEngine.git
    cd DSEngine
    pip install -r requirements.txt
    pip install -e .

Your First Experiment
---------------------
Create a minimal ``pipeline.yml``:

.. code-block:: yaml

    first_experiment:
      data:
        source: 'data/customers.csv'
      steps:
        - name: 'overview'
          type: 'summary'
          columns: []
          params: {}

Run it:

.. code-block:: bash

    python examples/run_pipeline.py --experiment first_experiment

Understanding the Output
------------------------
Check the ``outputs/`` folder for ``report.json``, saved plots, and a consolidated ``report.html``.

Adding More Experiments
-----------------------
You can add another top-level experiment config block inside the same YAML file.

API Reference
-------------
See the :doc:`api/exploration` and other API pages for details on supported blocks.
