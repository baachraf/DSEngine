"""
ds_engine.reporting.html_report
===============================

Generate a self-contained HTML report with Jinja2 combining stats and embedded plots.
"""

import os
import io
import base64
from pathlib import Path
import matplotlib.figure
from jinja2 import Template
from typing import Any, Optional
from ds_engine.utils import plot_utils, logger

# The template must be hardcoded here instead of a separate file as per guidelines Section 5.3
TEMPLATE_STR = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DSEngine Report: {{ experiment_name }}</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
               line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }
        .header { background-color: #fff; padding: 20px 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 20px; border-top: 5px solid #2E75B6; }
        .header h1 { margin-top: 0; color: #1a1a1a; font-weight: 600; }
        .meta-stamp { color: #666; font-size: 0.95em; }
        .warnings-box { background-color: #fff3cd; color: #856404; padding: 15px 20px; border-radius: 6px; border-left: 4px solid #ffeeba; margin-bottom: 20px; }
        .warnings-box h3 { margin-top: 0; margin-bottom: 10px; }
        .step-section { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 30px; }
        .step-title { margin-top: 0; border-bottom: 2px solid #eee; padding-bottom: 10px; color: #2E75B6; }
        .dict-table { width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 20px; font-size: 0.95em; }
        .dict-table th, .dict-table td { border: 1px solid #e0e0e0; padding: 10px 15px; text-align: left; }
        .dict-table th { background-color: #f4f6f8; color: #444; font-weight: 600; width: 30%; }
        .dict-table tr:nth-child(even) { background-color: #fafafa; }
        .plot-container { text-align: center; margin-top: 20px; }
        .plot-container img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 15px; }
        .footer { text-align: center; padding: 20px; color: #888; font-size: 0.9em; border-top: 1px solid #ddd; margin-top: 40px; }
        pre { background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; font-size: 0.9em; }
    </style>
</head>
<body>

    <div class="header">
        <h1>{{ experiment_name }}</h1>
        <div class="meta-stamp">
            <strong>Run Timestamp:</strong> {{ timestamp }} &nbsp;|&nbsp;
            <strong>Rows:</strong> {{ inspection.shape.rows }} &nbsp;|&nbsp;
            <strong>Columns:</strong> {{ inspection.shape.columns }}
        </div>
    </div>

    {% if inspection.warning_count > 0 %}
    <div class="warnings-box">
        <h3>Pre-flight Warnings ({{ inspection.warning_count }})</h3>
        <ul>
            {% for w in inspection.warnings %}
            <li><strong>{{ w.column }}</strong>: {{ w.message }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

    {% for step_name, data in steps.items() %}
    {% if step_name != 'pre_flight_inspection' %}
    <div class="step-section">
        <h2 class="step-title">Step: {{ step_name }}</h2>
        
        {% if data.status == 'FAILED' %}
            <div class="warnings-box" style="background-color: #f8d7da; color: #721c24; border-color: #f5c6cb;">
                <strong>FAILED:</strong> <pre>{{ data.error }}</pre>
            </div>
        {% else %}
            <table class="dict-table">
                <tbody>
                    {% for k, v in data.items() %}
                    <tr>
                        <th>{{ k }}</th>
                        <td>
                            {% if v is mapping or v is iterable and v is not string %}
                                <pre>{{ v | tojson(indent=2) }}</pre>
                            {% else %}
                                {{ v }}
                            {% endif %}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        {% endif %}

        {% if step_name in base64_plots and base64_plots[step_name] %}
        <div class="plot-container">
            {% for b64 in base64_plots[step_name] %}
            <img src="data:image/png;base64,{{ b64 }}" alt="Plot for {{ step_name }}">
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endif %}
    {% endfor %}

    <div class="footer">
        DSEngine Pipeline Report &bull; Generated automatically &bull; Output Directory: {{ output_dir }}
    </div>

</body>
</html>"""

def export(
    results: dict[str, dict[str, Any]], 
    figures: dict[str, list[matplotlib.figure.Figure]],
    output_dir: str, 
    experiment_name: str,
    saved_plot_paths: Optional[dict[str, list[str]]] = None
) -> str:
    """Generate self-contained HTML report.
    
    Args:
        results (dict): All output_data from steps.
        figures (dict): Dictionary of step_name to list of matplotlib Figures.
        output_dir (str): output directory path.
        experiment_name (str): The name of the experiment.
        saved_plot_paths (dict): Optional, ignored because we embed base64 directly
                                 to guarantee the HTML is strictly self-contained.
                                 
    Returns:
        str: Absolute path to the saved HTML file.
    """
    timestamp = results.get('pre_flight_inspection', {}).get('run_timestamp', '')
    inspection = results.get('pre_flight_inspection', {})
    
    base64_plots: dict[str, list[str]] = {}
    
    for step_name, figs in figures.items():
        step_b64 = []
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=plot_utils.EMBED_DPI, bbox_inches='tight')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            step_b64.append(b64)
        base64_plots[step_name] = step_b64
        
    template = Template(TEMPLATE_STR)
    html_content = template.render(
        experiment_name=experiment_name,
        timestamp=timestamp,
        inspection=inspection,
        steps=results,
        base64_plots=base64_plots,
        output_dir=os.path.relpath(output_dir) if os.path.isabs(output_dir) else output_dir
    )
    
    out_path = Path(output_dir) / 'report.html'
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return str(out_path)
    except Exception as e:
        logger.log_error(f"Failed to write report.html: {e}")
        return ""
