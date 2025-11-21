
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from .results import ExperimentResults


# ============================================================================
# JSON Export
# ============================================================================

def export_json(results: ExperimentResults, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        'timestamp': results.timestamp,
        'description': results.description,
        'objects': results.list_objects(),
        'methods': results.list_methods(),
        'results': {}
    }

    for obj_name, methods in results.results.items():
        export_data['results'][obj_name] = {}
        for method_name, result in methods.items():
            export_data['results'][obj_name][method_name] = result.to_dict(include_arrays=False)

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Results exported to {output_path}")


def export_summary_csv(results: ExperimentResults, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = results.summary_table()
    summary_df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"Summary statistics saved to {output_path}")


def export_per_object_csv(results: ExperimentResults, output_path: Path):
    """Export per-object results to CSV.

    Args:
        results: ExperimentResults object
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_object_df = results.per_object_table()
    per_object_df.to_csv(output_path, index=False, float_format='%.4f')

    print(f"Per-object results saved to {output_path}")


def export_latex_summary(results: ExperimentResults, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = results.summary_table()

    # Select key columns for publication
    display_cols = ['Method', 'Mean AUROC', 'Std AUROC', 'Mean AU-PRO', 'Std AU-PRO']
    display_cols = [col for col in display_cols if col in summary_df.columns]
    table_df = summary_df[display_cols]

    latex_str = table_df.to_latex(
        index=False,
        float_format='%.4f',
        caption='Summary of anomaly detection performance across all objects',
        label='tab:summary',
        position='htbp'
    )

    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"LaTeX summary table saved to {output_path}")


def export_latex_per_object(results: ExperimentResults, output_path: Path, metric: str = 'AUROC'):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_object_df = results.per_object_table()

    # Pivot to show objects as rows, methods as columns
    pivot_df = per_object_df.pivot_table(
        index='Object',
        columns='Method',
        values=metric
    )

    latex_str = pivot_df.to_latex(
        float_format='%.4f',
        caption=f'{metric} scores by object and method',
        label=f'tab:per_object_{metric.lower()}',
        position='htbp'
    )

    with open(output_path, 'w') as f:
        f.write(latex_str)

    print(f"LaTeX per-object table saved to {output_path}")


def generate_markdown_report(results: ExperimentResults, output_path: Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = results.summary_table()
    per_object_df = results.per_object_table()

    lines = []
    lines.append("# Anomaly Detection Evaluation Report\n")
    lines.append(f"**Generated:** {results.timestamp}\n")

    if results.description:
        lines.append(f"**Description:** {results.description}\n")

    lines.append(f"**Objects evaluated:** {len(results.list_objects())}\n")
    lines.append(f"**Methods evaluated:** {', '.join(results.list_methods())}\n")
    lines.append("\n---\n")

    # Summary statistics
    lines.append("## Summary Statistics\n")
    lines.append("Mean performance across all objects:\n")
    lines.append(summary_df.to_markdown(index=False, floatfmt='.4f'))
    lines.append("\n")

    # Best performing method
    best_method = summary_df.iloc[0]
    lines.append(f"**Best method:** {best_method['Method']} ")
    lines.append(f"(AUROC: {best_method['Mean AUROC']:.4f} ± {best_method['Std AUROC']:.4f})\n")
    lines.append("\n---\n")

    # Per-object breakdown
    lines.append("## Per-Object Results\n")

    # Pivot for better display
    for metric in ['AUROC', 'AU-PRO', 'IoU']:
        if metric not in per_object_df.columns:
            continue

        lines.append(f"### {metric}\n")
        pivot = per_object_df.pivot_table(
            index='Object',
            columns='Method',
            values=metric
        )
        lines.append(pivot.to_markdown(floatfmt='.4f'))
        lines.append("\n")

    # Challenging objects
    lines.append("---\n")
    lines.append("## Analysis\n")

    # Find most/least challenging objects (based on best method's AUROC)
    best_method_name = summary_df.iloc[0]['Method']
    method_df = per_object_df[per_object_df['Method'] == best_method_name].copy()
    method_df = method_df.sort_values('AUROC')

    lines.append(f"### Most Challenging Objects (for {best_method_name})\n")
    worst_3 = method_df.head(3)
    for _, row in worst_3.iterrows():
        lines.append(f"- **{row['Object']}**: AUROC = {row['AUROC']:.4f}\n")

    lines.append(f"\n### Easiest Objects (for {best_method_name})\n")
    best_3 = method_df.tail(3)
    for _, row in best_3.iterrows():
        lines.append(f"- **{row['Object']}**: AUROC = {row['AUROC']:.4f}\n")

    # Write report
    with open(output_path, 'w') as f:
        f.writelines(lines)

    print(f"Markdown report saved to {output_path}")



def export_all(results: ExperimentResults, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting results in all formats...")

    # JSON
    export_json(results, output_dir / 'results.json')

    # CSV
    export_summary_csv(results, output_dir / 'summary.csv')
    export_per_object_csv(results, output_dir / 'per_object.csv')

    # LaTeX
    latex_dir = output_dir / 'latex'
    latex_dir.mkdir(exist_ok=True)
    export_latex_summary(results, latex_dir / 'summary.tex')
    export_latex_per_object(results, latex_dir / 'per_object_auroc.tex', metric='AUROC')
    export_latex_per_object(results, latex_dir / 'per_object_aupro.tex', metric='AU-PRO')

    # Markdown
    generate_markdown_report(results, output_dir / 'report.md')

    print(f"\nAll exports completed! Results saved to: {output_dir}")
