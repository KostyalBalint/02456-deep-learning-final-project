import json
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd


@dataclass
class MethodResults:
    method_name: str
    object_name: str
    scores: np.ndarray
    maps: List[np.ndarray]
    metrics: Dict[str, float]
    config: Dict[str, Any] = field(default_factory=dict)
    num_features: Optional[int] = None

    def to_dict(self, include_arrays=False) -> Dict:
        result = {
            'method_name': self.method_name,
            'object_name': self.object_name,
            'metrics': self.metrics,
            'config': self.config,
            'num_features': self.num_features
        }

        if include_arrays:
            result['scores'] = self.scores.tolist()

        return result

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata as JSON
        meta_path = path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(self.to_dict(include_arrays=False), f, indent=2)

        # Save arrays as NPZ
        array_path = path.with_suffix('.npz')        
        np.savez_compressed(
            array_path,
            scores=self.scores,
            **{f'map_{i}': m for i, m in enumerate(self.maps)}
        )


    @classmethod
    def load(cls, path: Path) -> "MethodResults":
        path = Path(path)

        meta_path = path.with_suffix('.json')
        with open(meta_path, 'r') as f:
            data = json.load(f)

        array_path = path.with_suffix('.npz')
        arrays = np.load(array_path)
        scores = arrays['scores']

        map_keys = sorted(
            [k for k in arrays.keys() if k.startswith("map_")],
            key=lambda x: int(x.split("_")[1]),
        )
        maps = [arrays[k] for k in map_keys]

        pred_keys = sorted(
            [k for k in arrays.keys() if k.startswith("pred_")],
            key=lambda x: int(x.split("_")[1]),
        )

        return cls(
            method_name=data['method_name'],
            object_name=data['object_name'],
            scores=scores,
            maps=maps,
            metrics=data['metrics'],
            config=data.get('config', {}),
            num_features=data.get('num_features')
        )



@dataclass
class ExperimentResults:
    results: Dict[str, Dict[str, MethodResults]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    description: str = ""

    def get_method(self, method_name: str) -> Dict[str, MethodResults]:
        return {
            obj: methods[method_name]
            for obj, methods in self.results.items()
            if method_name in methods
        }

    def get_object(self, object_name: str) -> Dict[str, MethodResults]:
        return self.results.get(object_name, {})

    def list_methods(self) -> List[str]:
        methods = set()
        for obj_results in self.results.values():
            methods.update(obj_results.keys())
        return sorted(list(methods))

    def list_objects(self) -> List[str]:
        return sorted(list(self.results.keys()))

    def summary_table(self) -> pd.DataFrame:
        summary_data = []

        for method_name in self.list_methods():
            method_results = self.get_method(method_name)

            # Collect metrics across objects
            auroc_scores = []
            aupro_scores = []
            pixel_auroc_scores = []
            pixel_f1_scores = []
            iou_scores = []

            for obj, result in method_results.items():
                metrics = result.metrics
                auroc_scores.append(metrics.get('auroc', np.nan))

                # Only include non-NaN values for pixel metrics
                if not np.isnan(metrics.get('aupro', np.nan)):
                    aupro_scores.append(metrics['aupro'])
                if not np.isnan(metrics.get('pixel_auroc', np.nan)):
                    pixel_auroc_scores.append(metrics['pixel_auroc'])
                if not np.isnan(metrics.get('pixel_f1', np.nan)):
                    pixel_f1_scores.append(metrics['pixel_f1'])
                if not np.isnan(metrics.get('mean_iou', np.nan)):
                    iou_scores.append(metrics['mean_iou'])

            summary_data.append({
                'Method': method_name,
                'Mean AUROC': np.mean(auroc_scores) if auroc_scores else np.nan,
                'Std AUROC': np.std(auroc_scores) if auroc_scores else np.nan,
                'Mean AU-PRO': np.mean(aupro_scores) if aupro_scores else np.nan,
                'Std AU-PRO': np.std(aupro_scores) if aupro_scores else np.nan,
                'Mean Pixel AUROC': np.mean(pixel_auroc_scores) if pixel_auroc_scores else np.nan,
                'Mean Pixel F1': np.mean(pixel_f1_scores) if pixel_f1_scores else np.nan,
                'Mean IoU': np.mean(iou_scores) if iou_scores else np.nan,
                'Num Objects': len(method_results)
            })

        df = pd.DataFrame(summary_data)
        return df.sort_values('Mean AUROC', ascending=False)

    def per_object_table(self) -> pd.DataFrame:
        rows = []

        for obj_name in self.list_objects():
            for method_name, result in self.results[obj_name].items():
                rows.append({
                    'Object': obj_name,
                    'Method': method_name,
                    'AUROC': result.metrics.get('auroc', np.nan),
                    'AU-PRO': result.metrics.get('aupro', np.nan),
                    'Pixel AUROC': result.metrics.get('pixel_auroc', np.nan),
                    'Pixel F1': result.metrics.get('pixel_f1', np.nan),
                    'IoU': result.metrics.get('mean_iou', np.nan)
                })

        return pd.DataFrame(rows)

    def compare_methods(self, method_a: str, method_b: str, metric: str = 'auroc') -> pd.DataFrame:
        results_a = self.get_method(method_a)
        results_b = self.get_method(method_b)

        # Find common objects
        common_objects = set(results_a.keys()) & set(results_b.keys())

        comparison_data = []
        for obj in sorted(common_objects):
            val_a = results_a[obj].metrics.get(metric, np.nan)
            val_b = results_b[obj].metrics.get(metric, np.nan)
            diff = val_b - val_a if not (np.isnan(val_a) or np.isnan(val_b)) else np.nan

            comparison_data.append({
                'Object': obj,
                f'{method_a}': val_a,
                f'{method_b}': val_b,
                'Difference (B-A)': diff,
                'Improvement %': (diff / val_a * 100) if val_a != 0 else np.nan
            })

        df = pd.DataFrame(comparison_data)

        # Add summary statistics
        if len(df) > 0:
            values_a = df[method_a].dropna()
            values_b = df[method_b].dropna()

            if len(values_a) > 0 and len(values_b) > 0:
                from scipy import stats
                t_stat, p_value = stats.ttest_rel(values_a, values_b)

                print(f"\nStatistical Comparison ({metric}):")
                print(f"  {method_a}: {values_a.mean():.4f} ± {values_a.std():.4f}")
                print(f"  {method_b}: {values_b.mean():.4f} ± {values_b.std():.4f}")
                print(f"  Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")
                print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")

        return df

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            'timestamp': self.timestamp,
            'description': self.description,
            'objects': self.list_objects(),
            'methods': self.list_methods()
        }

        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        for obj_name, methods in self.results.items():
            for method_name, result in methods.items():
                result_path = path / f"{obj_name}_{method_name}"
                result.save(result_path)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentResults':
        path = Path(path)

        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load all method results
        results = {}
        for obj_name in metadata['objects']:
            results[obj_name] = {}
            for method_name in metadata['methods']:
                result_path = path / f"{obj_name}_{method_name}"
                if result_path.with_suffix('.json').exists():
                    results[obj_name][method_name] = MethodResults.load(result_path)

        return cls(
            results=results,
            timestamp=metadata['timestamp'],
            description=metadata.get('description', '')
        )

    def merge(self, other: 'ExperimentResults') -> 'ExperimentResults':
        """Merge results from another ExperimentResults into this one."""
        merged_results = {}

        # Get all objects from both results
        all_objects = set(self.list_objects()) | set(other.list_objects())

        for obj in all_objects:
            merged_results[obj] = {}

            # Add methods from self
            if obj in self.results:
                for method_name, result in self.results[obj].items():
                    merged_results[obj][method_name] = result

            # Add methods from other
            if obj in other.results:
                for method_name, result in other.results[obj].items():
                    merged_results[obj][method_name] = result

        return ExperimentResults(
            results=merged_results,
            description=f"{self.description} + {other.description}"
        )
