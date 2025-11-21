import json
from pathlib import Path
from .results import ExperimentResults


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


def export_all(results: ExperimentResults, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting results in all formats...")

    # JSON
    export_json(results, output_dir / 'results.json')

    # CSV
    export_summary_csv(results, output_dir / 'summary.csv')
    export_per_object_csv(results, output_dir / 'per_object.csv')

    print(f"\nAll exports completed! Results saved to: {output_dir}")
