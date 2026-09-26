"""
MLForge - Data Quality & ML Readiness Audit Module
=========================================================
Provides comprehensive diagnostic dataset audits for tabular machine learning:
- Column intelligence (numeric, categorical, datetime, ID-like, high cardinality)
- Data hygiene (missing values, duplicates, infinite values, near-constants, empty columns)
- Outlier detection (IQR method)
- Target analysis & data leakage detection (classification & regression)
- Deterministic, explainable ML Readiness Score (0-100)
- Actionable recommendations and severity-coded issue tracking
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import numpy as np
import pandas as pd


# =====================================================================
# DATA STRUCTURES
# =====================================================================

@dataclass
class AuditIssue:
    """Represents a single diagnostic finding."""
    severity: str  # "Critical" | "High" | "Medium" | "Low" | "Info"
    category: str  # "Missing Data" | "Duplicates" | "Column Quality" | "Outliers" | "Target" | "Data Leakage" | "Dataset Shape" | "Data Types"
    message: str
    recommendation: str
    column: Optional[Union[str, List[str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreDeduction:
    """Represents an itemized penalty contributing to the ML Readiness Score."""
    category: str
    penalty: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColumnProfile:
    """In-depth profile for an individual column."""
    name: str
    detected_type: str  # "numeric_int", "numeric_float", "categorical", "boolean", "datetime", "unknown"
    dtype: str
    total_count: int
    null_count: int
    null_pct: float
    unique_count: int
    unique_ratio: float
    is_constant: bool = False
    is_near_constant: bool = False
    is_id_like: bool = False
    id_reason: Optional[str] = None
    is_high_cardinality: bool = False
    is_empty: bool = False
    infinite_count: int = 0
    outlier_count: int = 0
    outlier_pct: float = 0.0
    outlier_lower_bound: Optional[float] = None
    outlier_upper_bound: Optional[float] = None
    top_value: Any = None
    top_frequency: Optional[int] = None
    top_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetAnalysis:
    """Target column analysis for supervised learning."""
    target_col: str
    task_type: str  # "classification" | "regression"
    total_count: int
    valid_count: int
    null_count: int
    null_pct: float
    infinite_count: int = 0
    is_constant: bool = False
    n_classes: Optional[int] = None
    class_distribution: Optional[Dict[str, int]] = None
    class_percentages: Optional[Dict[str, float]] = None
    minority_class: Optional[str] = None
    minority_pct: Optional[float] = None
    imbalance_ratio: Optional[float] = None
    is_severely_imbalanced: bool = False
    is_moderately_imbalanced: bool = False
    leakage_suspects: List[Dict[str, Any]] = field(default_factory=list)
    regression_stats: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditResult:
    """Comprehensive structured audit output."""
    score: int
    grade: str
    severity: str  # Overall highest severity
    dataset_stats: Dict[str, Any]
    detected_issues: List[AuditIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)
    classified_columns: Dict[str, List[str]] = field(default_factory=dict)
    outlier_summary: Dict[str, Any] = field(default_factory=dict)
    target_analysis: Optional[TargetAnalysis] = None
    score_breakdown: List[ScoreDeduction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "severity": self.severity,
            "dataset_stats": self.dataset_stats,
            "detected_issues": [issue.to_dict() for issue in self.detected_issues],
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "column_profiles": {k: v.to_dict() for k, v in self.column_profiles.items()},
            "classified_columns": self.classified_columns,
            "outlier_summary": self.outlier_summary,
            "target_analysis": self.target_analysis.to_dict() if self.target_analysis else None,
            "score_breakdown": [d.to_dict() for d in self.score_breakdown]
        }


# =====================================================================
# AUDIT ENGINE
# =====================================================================

class DataAuditor:
    """
    Core engine for auditing dataset quality and machine learning readiness.
    """

    ID_PATTERNS = re.compile(r"(^|_)(id|uuid|identifier|pk|key|code|customer_id|user_id|account_id|client_id|txn_id|transaction_id)($|_)", re.IGNORECASE)
    UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
    PREFIXED_ID_REGEX = re.compile(r"^[a-z]{1,5}[-_]?\d{3,}$", re.IGNORECASE)
    DATE_PATTERNS = re.compile(
        r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"  # 2024-01-01 or 2024/01/01
        r"|^\d{1,2}[-/.]\d{1,2}[-/.]\d{4}"  # 01/01/2024
        r"|^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"  # ISO format
    )

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        """
        Initialize the auditor with a pandas DataFrame and optional target column.
        """
        self.df = df
        self.target_col = target_col if (target_col is not None and df is not None and target_col in df.columns) else None

    # -----------------------------------------------------------------
    # TYPE DETECTION & COLUMN INTELLIGENCE
    # -----------------------------------------------------------------

    def _is_datetime_series(self, series: pd.Series) -> bool:
        """Heuristically check if a series represents dates/times."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True

        # Only check string/object types
        if not pd.api.types.is_object_dtype(series) and not isinstance(series.dtype, pd.StringDtype):
            return False

        non_null = series.dropna().astype(str).str.strip()
        if len(non_null) == 0:
            return False

        # Take a representative sample (up to 30 items)
        sample = non_null.head(30)
        
        # Avoid purely numeric strings (like IDs, years, zip codes)
        if all(val.isdigit() and len(val) <= 6 for val in sample):
            return False

        # Pattern match check
        pattern_matches = sum(bool(self.DATE_PATTERNS.search(val)) for val in sample)
        if pattern_matches / len(sample) >= 0.7:
            # Confirm by attempting to parse with pandas
            try:
                parsed = pd.to_datetime(sample, errors="coerce")
                if parsed.notna().sum() / len(sample) >= 0.7:
                    return True
            except Exception:
                pass

        return False

    def _detect_column_type(self, series: pd.Series) -> str:
        """Classify column into primary diagnostic type."""
        if self._is_datetime_series(series):
            return "datetime"
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_integer_dtype(series):
            return "numeric_int"
        if pd.api.types.is_float_dtype(series):
            return "numeric_float"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric_float"
        return "categorical"

    def _check_id_characteristics(self, col_name: str, series: pd.Series, unique_ratio: float, n_rows: int) -> Tuple[bool, Optional[str]]:
        """Determine if a column behaves like an artificial identifier."""
        name_match = bool(self.ID_PATTERNS.search(col_name))
        non_null = series.dropna()
        n_valid = len(non_null)

        if n_valid < 5:
            return False, None

        # Continuous floating point numbers with arbitrary decimals are NOT ID columns
        if pd.api.types.is_float_dtype(non_null) and not name_match:
            return False, None

        # Rule 1: Explicit ID name pattern and high uniqueness (>= 80%)
        if name_match and unique_ratio >= 0.80:
            return True, f"Column name matches identifier pattern ('{col_name}') and {unique_ratio*100:.1f}% of values are unique."

        # Rule 2: Sequential increasing integer index or key
        if pd.api.types.is_integer_dtype(non_null) and n_valid >= 20:
            try:
                if non_null.is_monotonic_increasing:
                    if name_match or (non_null.iloc[-1] - non_null.iloc[0] == n_valid - 1):
                        return True, "Monotonically increasing sequential integers indicating index or surrogate key."
                    if unique_ratio >= 0.99:
                        return True, f"Strictly increasing unique integer sequence ({unique_ratio*100:.1f}% unique)."
            except Exception:
                pass

        # Rule 3: String column matching standard UUID or prefixed ID pattern (e.g. 'CUST_0001')
        if (pd.api.types.is_object_dtype(non_null) or isinstance(non_null.dtype, pd.StringDtype)) and n_valid >= 10:
            sample_str = non_null.head(20).astype(str)
            if any(self.UUID_REGEX.match(s) for s in sample_str):
                return True, "Values follow standard UUID format."
            if all(self.PREFIXED_ID_REGEX.match(s) for s in sample_str):
                return True, "Values follow structured alphanumeric identifier pattern (e.g. 'CUST_0001')."

        return False, None

    # -----------------------------------------------------------------
    # OUTLIER DETECTION (IQR)
    # -----------------------------------------------------------------

    def _compute_iqr_outliers(self, series: pd.Series) -> Tuple[int, float, Optional[float], Optional[float]]:
        """Compute outliers using the standard Interquartile Range method."""
        non_null = series.dropna()
        # Filter finite values only
        valid = non_null[np.isfinite(non_null)] if pd.api.types.is_numeric_dtype(non_null) else pd.Series(dtype=float)
        
        if len(valid) < 8:
            return 0, 0.0, None, None

        q1 = float(valid.quantile(0.25))
        q3 = float(valid.quantile(0.75))
        iqr = q3 - q1

        if iqr <= 0:
            return 0, 0.0, None, None

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = (valid < lower_bound) | (valid > upper_bound)
        count = int(outliers.sum())
        pct = round((count / len(valid)) * 100, 2)

        return count, pct, lower_bound, upper_bound

    # -----------------------------------------------------------------
    # TARGET ANALYSIS & LEAKAGE DETECTION
    # -----------------------------------------------------------------

    def _analyze_target(self, target_col: str, col_profiles: Dict[str, ColumnProfile]) -> TargetAnalysis:
        """Perform deep audit on the designated prediction target."""
        series = self.df[target_col]
        total_count = len(series)
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_count) * 100, 2) if total_count > 0 else 100.0
        valid_series = series.dropna()
        valid_count = len(valid_series)

        inf_count = 0
        if pd.api.types.is_numeric_dtype(series):
            inf_count = int(np.isinf(valid_series).sum())

        is_const = (valid_series.nunique() <= 1) if valid_count > 0 else True

        # Infer problem type
        task_type = "classification"
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            unique_count = valid_series.nunique()
            # Classification if small unique set or integer labels
            if unique_count > max(10, int(0.05 * valid_count)):
                task_type = "regression"

        # Classification metrics
        n_classes = None
        class_dist = None
        class_pcts = None
        minority_class = None
        minority_pct = None
        imbalance_ratio = None
        is_sev_imb = False
        is_mod_imb = False

        # Regression metrics
        reg_stats = None

        if task_type == "classification" and valid_count > 0:
            vc = valid_series.value_counts()
            n_classes = int(len(vc))
            class_dist = {str(k): int(v) for k, v in vc.items()}
            class_pcts = {str(k): round((v / valid_count) * 100, 2) for k, v in vc.items()}
            
            if n_classes > 1:
                min_c = vc.idxmin()
                max_c = vc.idxmax()
                minority_class = str(min_c)
                min_val = vc[min_c]
                max_val = vc[max_c]
                minority_pct = round((min_val / valid_count) * 100, 2)
                imbalance_ratio = round(max_val / max(1, min_val), 2)

                if minority_pct < 5.0 or imbalance_ratio >= 10.0:
                    is_sev_imb = True
                elif minority_pct < 15.0 or imbalance_ratio >= 4.0:
                    is_mod_imb = True
        elif task_type == "regression" and valid_count > 0:
            finite_valid = valid_series[np.isfinite(valid_series)]
            if len(finite_valid) > 0:
                reg_stats = {
                    "mean": round(float(finite_valid.mean()), 4),
                    "std": round(float(finite_valid.std()), 4) if len(finite_valid) > 1 else 0.0,
                    "min": round(float(finite_valid.min()), 4),
                    "max": round(float(finite_valid.max()), 4),
                    "skewness": round(float(finite_valid.skew()), 4) if len(finite_valid) > 2 else 0.0
                }

        # Data Leakage Detection
        leakage_suspects = []
        for other_col in self.df.columns:
            if other_col == target_col:
                continue

            other_series = self.df[other_col]
            
            # Check 1: Exact equivalence
            try:
                if series.equals(other_series):
                    leakage_suspects.append({
                        "column": other_col,
                        "type": "exact_duplicate",
                        "correlation": 1.0,
                        "reason": f"Column '{other_col}' is identical to target column '{target_col}'."
                    })
                    continue
            except Exception:
                pass

            # Check 2: High Pearson correlation for numeric pairs
            if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(other_series):
                try:
                    common = self.df[[target_col, other_col]].dropna()
                    # Filter finite values
                    finite_mask = np.isfinite(common[target_col]) & np.isfinite(common[other_col])
                    common = common[finite_mask]
                    if len(common) >= 10:
                        std_t = float(common[target_col].std())
                        std_o = float(common[other_col].std())
                        if std_t > 1e-9 and std_o > 1e-9:
                            corr = float(common[target_col].corr(common[other_col]))
                            if not np.isnan(corr) and abs(corr) >= 0.99:
                                leakage_suspects.append({
                                    "column": other_col,
                                    "type": "high_correlation",
                                    "correlation": round(corr, 4),
                                    "reason": f"Extreme correlation (|r| = {abs(corr):.4f} >= 0.99) with target column '{target_col}'."
                                })
                except Exception:
                    pass

        return TargetAnalysis(
            target_col=target_col,
            task_type=task_type,
            total_count=total_count,
            valid_count=valid_count,
            null_count=null_count,
            null_pct=null_pct,
            infinite_count=inf_count,
            is_constant=is_const,
            n_classes=n_classes,
            class_distribution=class_dist,
            class_percentages=class_pcts,
            minority_class=minority_class,
            minority_pct=minority_pct,
            imbalance_ratio=imbalance_ratio,
            is_severely_imbalanced=is_sev_imb,
            is_moderately_imbalanced=is_mod_imb,
            leakage_suspects=leakage_suspects,
            regression_stats=reg_stats
        )

    # -----------------------------------------------------------------
    # DUPLICATE & NEAR-DUPLICATE COLUMNS
    # -----------------------------------------------------------------

    def _find_duplicate_columns(self) -> List[Tuple[str, str, float]]:
        """Identify pairs of identical or near-duplicate columns."""
        dup_pairs = []
        cols = list(self.df.columns)
        n_cols = len(cols)
        
        # Limit comparison to reasonable column count to maintain high performance
        max_cols_to_check = min(n_cols, 100)
        check_cols = cols[:max_cols_to_check]

        for i in range(len(check_cols)):
            col1 = check_cols[i]
            s1 = self.df[col1]
            for j in range(i + 1, len(check_cols)):
                col2 = check_cols[j]
                s2 = self.df[col2]

                try:
                    if s1.equals(s2):
                        dup_pairs.append((col1, col2, 100.0))
                    elif len(s1) >= 20 and s1.dtype == s2.dtype:
                        # Check match ratio on valid elements
                        match_ratio = (s1 == s2).mean()
                        if match_ratio >= 0.99:
                            dup_pairs.append((col1, col2, round(match_ratio * 100, 2)))
                except Exception:
                    continue

        return dup_pairs

    # -----------------------------------------------------------------
    # AUDIT EXECUTION
    # -----------------------------------------------------------------

    def audit(self) -> AuditResult:
        """Run complete audit and return structured diagnostic results."""
        df = self.df

        # --- Handle empty dataframe case ---
        if df is None or df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            n_rows = 0 if df is None else df.shape[0]
            n_cols = 0 if df is None else df.shape[1]
            issues = [
                AuditIssue(
                    severity="Critical",
                    category="Dataset Shape",
                    message=f"Dataset contains {n_rows} rows and {n_cols} columns. No data available for modeling.",
                    recommendation="Upload a valid dataset with at least 20 rows and 2 feature columns."
                )
            ]
            deductions = [
                ScoreDeduction(category="Dataset Shape", penalty=100.0, reason="Empty dataset with 0 usable rows/columns.")
            ]
            return AuditResult(
                score=0,
                grade="Critical (Unusable)",
                severity="Critical",
                dataset_stats={
                    "rows": n_rows,
                    "columns": n_cols,
                    "total_cells": 0,
                    "missing_cells": 0,
                    "missing_pct": 0.0,
                    "duplicate_rows": 0,
                    "duplicate_pct": 0.0,
                    "numeric_cols_count": 0,
                    "categorical_cols_count": 0,
                    "datetime_cols_count": 0,
                    "boolean_cols_count": 0,
                    "unusable_cols_count": 0
                },
                detected_issues=issues,
                warnings=["Dataset is empty or has 0 columns."],
                recommendations=["Please upload a valid structured dataset."],
                score_breakdown=deductions
            )

        n_rows, n_cols = df.shape
        total_cells = int(df.size)
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0.0

        dup_rows = int(df.duplicated().sum())
        dup_pct = round((dup_rows / n_rows) * 100, 2) if n_rows > 0 else 0.0

        # Data collection containers
        issues: List[AuditIssue] = []
        recommendations: List[str] = []
        warnings: List[str] = []
        col_profiles: Dict[str, ColumnProfile] = {}
        classified_cols: Dict[str, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "boolean": [],
            "id_like": [],
            "high_cardinality": [],
            "constant": [],
            "near_constant": [],
            "empty": [],
            "high_missing": [],
            "outliers": [],
            "infinite": [],
            "unusable": []
        }

        # -------------------------------------------------------------
        # 1. COLUMN LEVEL AUDIT
        # -------------------------------------------------------------
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_pct = round((null_count / n_rows) * 100, 2)
            non_null = series.dropna()
            n_valid = len(non_null)
            unique_count = int(non_null.nunique())
            unique_ratio = round(unique_count / n_valid, 4) if n_valid > 0 else 0.0

            # Type classification
            col_type = self._detect_column_type(series)

            # Check special characteristics
            is_empty = (null_count == n_rows)
            is_constant = (unique_count <= 1 and not is_empty)
            
            # Near-constant check (> 95% single value)
            is_near_constant = False
            top_val = None
            top_freq = None
            top_ratio = None
            if n_valid > 0:
                try:
                    vc = non_null.value_counts()
                    if not vc.empty:
                        top_val = vc.index[0]
                        top_freq = int(vc.iloc[0])
                        top_ratio = round((top_freq / n_valid) * 100, 2)
                        if top_ratio >= 95.0 and not is_constant:
                            is_near_constant = True
                except Exception:
                    pass

            # ID-like detection
            is_id, id_reason = self._check_id_characteristics(str(col), series, unique_ratio, n_rows)

            # High cardinality detection for non-numeric/categorical
            is_high_card = False
            if col_type in ("categorical", "unknown") and not is_constant:
                if unique_count > 50 or (n_rows >= 30 and unique_ratio >= 0.25 and unique_count > 10):
                    is_high_card = True

            # Outlier & Infinite checks for numeric
            outlier_count, outlier_pct, out_low, out_high = 0, 0.0, None, None
            inf_count = 0
            if "numeric" in col_type:
                try:
                    inf_count = int(np.isinf(non_null).sum())
                except Exception:
                    inf_count = 0
                outlier_count, outlier_pct, out_low, out_high = self._compute_iqr_outliers(series)

            profile = ColumnProfile(
                name=str(col),
                detected_type=col_type,
                dtype=str(series.dtype),
                total_count=n_rows,
                null_count=null_count,
                null_pct=null_pct,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                is_constant=is_constant,
                is_near_constant=is_near_constant,
                is_id_like=is_id,
                id_reason=id_reason,
                is_high_cardinality=is_high_card,
                is_empty=is_empty,
                infinite_count=inf_count,
                outlier_count=outlier_count,
                outlier_pct=outlier_pct,
                outlier_lower_bound=out_low,
                outlier_upper_bound=out_high,
                top_value=str(top_val) if top_val is not None else None,
                top_frequency=top_freq,
                top_ratio=top_ratio
            )
            col_profiles[str(col)] = profile

            # Classify into groups
            if "numeric" in col_type:
                classified_cols["numeric"].append(str(col))
            elif col_type == "categorical":
                classified_cols["categorical"].append(str(col))
            elif col_type == "datetime":
                classified_cols["datetime"].append(str(col))
            elif col_type == "boolean":
                classified_cols["boolean"].append(str(col))

            if is_id:
                classified_cols["id_like"].append(str(col))
            if is_high_card:
                classified_cols["high_cardinality"].append(str(col))
            if is_constant:
                classified_cols["constant"].append(str(col))
            if is_near_constant:
                classified_cols["near_constant"].append(str(col))
            if is_empty:
                classified_cols["empty"].append(str(col))
            if null_pct >= 50.0 and not is_empty:
                classified_cols["high_missing"].append(str(col))
            if outlier_count > 0:
                classified_cols["outliers"].append(str(col))
            if inf_count > 0:
                classified_cols["infinite"].append(str(col))
            if is_empty or is_constant or null_pct >= 90.0:
                classified_cols["unusable"].append(str(col))

        # -------------------------------------------------------------
        # 2. DETECT STRUCTURAL & SHAPE ISSUES
        # -------------------------------------------------------------
        if n_rows < 20:
            issues.append(AuditIssue(
                severity="High",
                category="Dataset Shape",
                message=f"Extremely small sample size ({n_rows} rows). Cross-validation and ML models may severely overfit or fail.",
                recommendation="Gather more observations (aim for at least 100+ rows) to ensure statistical validity."
            ))
        elif n_rows < 50:
            issues.append(AuditIssue(
                severity="Medium",
                category="Dataset Shape",
                message=f"Small dataset ({n_rows} rows). Simpler models (e.g. Ridge, Logistic Regression, Naive Bayes) are recommended.",
                recommendation="Use k-fold cross validation and avoid complex tree ensembles or deep learning."
            ))

        if n_cols > n_rows and n_rows >= 10:
            issues.append(AuditIssue(
                severity="Medium",
                category="Dataset Shape",
                message=f"High-dimensional dataset: {n_cols} columns > {n_rows} rows (curse of dimensionality).",
                recommendation="Apply dimensionality reduction (PCA) or feature selection / L1 regularization."
            ))

        # -------------------------------------------------------------
        # 3. DETECT MISSING VALUES ISSUES
        # -------------------------------------------------------------
        if missing_pct >= 50.0:
            issues.append(AuditIssue(
                severity="High",
                category="Missing Data",
                message=f"Extreme overall missingness ({missing_pct}% of all data cells are null).",
                recommendation="Inspect data acquisition pipeline. Consider dropping columns with > 70% missingness."
            ))
        elif missing_pct >= 10.0:
            issues.append(AuditIssue(
                severity="Medium",
                category="Missing Data",
                message=f"Moderate overall missingness ({missing_pct}% of data cells are null).",
                recommendation="Apply median/mean imputation for numeric columns and mode/constant imputation for categoricals."
            ))
        elif missing_pct > 0.0:
            issues.append(AuditIssue(
                severity="Low",
                category="Missing Data",
                message=f"Minor missingness ({missing_pct}% of data cells).",
                recommendation="Use quick median/mode imputation or drop rows with missing values."
            ))

        for c_name in classified_cols["empty"]:
            issues.append(AuditIssue(
                severity="High",
                category="Column Quality",
                column=c_name,
                message=f"Column '{c_name}' is 100% empty (contains only NaN values).",
                recommendation=f"Drop column '{c_name}' prior to training as it provides zero informational value."
            ))

        for c_name in classified_cols["high_missing"]:
            p = col_profiles[c_name]
            issues.append(AuditIssue(
                severity="Medium",
                category="Missing Data",
                column=c_name,
                message=f"Column '{c_name}' has high missingness ({p.null_pct}% null).",
                recommendation=f"Consider dropping '{c_name}' or creating an indicator feature before imputing."
            ))

        # -------------------------------------------------------------
        # 4. DUPLICATE ROWS & COLUMNS
        # -------------------------------------------------------------
        if dup_rows > 0:
            sev = "High" if dup_pct >= 25.0 else "Medium" if dup_pct >= 5.0 else "Low"
            issues.append(AuditIssue(
                severity=sev,
                category="Duplicates",
                message=f"Detected {dup_rows:,} duplicate rows ({dup_pct}% of dataset).",
                recommendation="Remove redundant rows in Step 4 (Duplicate Handler) to prevent data leakage between train/test sets."
            ))

        dup_col_pairs = self._find_duplicate_columns()
        for col1, col2, match_pct in dup_col_pairs:
            issues.append(AuditIssue(
                severity="Medium",
                category="Duplicates",
                column=[col1, col2],
                message=f"Columns '{col1}' and '{col2}' are {match_pct}% identical.",
                recommendation=f"Drop one of the duplicate columns ('{col2}') to reduce multicollinearity."
            ))

        # -------------------------------------------------------------
        # 5. COLUMN QUALITY (Constants, Near-Constants, IDs, High Card)
        # -------------------------------------------------------------
        for c_name in classified_cols["constant"]:
            issues.append(AuditIssue(
                severity="Medium",
                category="Column Quality",
                column=c_name,
                message=f"Column '{c_name}' has zero variance (contains only 1 unique value: '{col_profiles[c_name].top_value}').",
                recommendation=f"Drop column '{c_name}'. Zero-variance features cannot contribute to predictive performance."
            ))

        for c_name in classified_cols["near_constant"]:
            p = col_profiles[c_name]
            issues.append(AuditIssue(
                severity="Low",
                category="Column Quality",
                column=c_name,
                message=f"Column '{c_name}' is near-constant ({p.top_ratio}% of values are '{p.top_value}').",
                recommendation=f"Review '{c_name}' — consider dropping or binarizing if the rare class has special predictive significance."
            ))

        for c_name in classified_cols["id_like"]:
            p = col_profiles[c_name]
            issues.append(AuditIssue(
                severity="Medium",
                category="Column Quality",
                column=c_name,
                message=f"Column '{c_name}' appears to be an identifier: {p.id_reason}",
                recommendation=f"Exclude '{c_name}' from model features to prevent memorization and overfitting."
            ))

        for c_name in classified_cols["high_cardinality"]:
            p = col_profiles[c_name]
            issues.append(AuditIssue(
                severity="Medium",
                category="Column Quality",
                column=c_name,
                message=f"Categorical column '{c_name}' has high cardinality ({p.unique_count} unique values).",
                recommendation=f"Avoid standard One-Hot Encoding on '{c_name}'. Use Target Encoding, Frequency Encoding, or drop."
            ))

        # -------------------------------------------------------------
        # 6. INFINITE & OUTLIER VALUES
        # -------------------------------------------------------------
        for c_name in classified_cols["infinite"]:
            p = col_profiles[c_name]
            issues.append(AuditIssue(
                severity="High",
                category="Data Types",
                column=c_name,
                message=f"Numeric column '{c_name}' contains {p.infinite_count} infinite values (+/- inf).",
                recommendation=f"Replace infinite values in '{c_name}' with NaN or column bounds before training."
            ))

        total_outlier_count = sum(p.outlier_count for p in col_profiles.values())
        if total_outlier_count > 0:
            outlier_cols = classified_cols["outliers"]
            max_outlier_pct = max((p.outlier_pct for p in col_profiles.values()), default=0.0)
            sev = "High" if max_outlier_pct >= 15.0 else "Medium" if max_outlier_pct >= 5.0 else "Low"
            issues.append(AuditIssue(
                severity=sev,
                category="Outliers",
                column=outlier_cols,
                message=f"Found {total_outlier_count:,} statistical outliers across {len(outlier_cols)} numeric column(s) using IQR (max {max_outlier_pct}% in a column).",
                recommendation="Inspect extreme values. Consider RobustScaler or Tree-based algorithms which are resilient to outliers."
            ))

        # -------------------------------------------------------------
        # 7. TARGET ANALYSIS (If Target Provided)
        # -------------------------------------------------------------
        target_analysis = None
        if self.target_col and self.target_col in df.columns:
            target_analysis = self._analyze_target(self.target_col, col_profiles)
            
            # Check target validity
            if target_analysis.null_count == target_analysis.total_count:
                issues.append(AuditIssue(
                    severity="Critical",
                    category="Target",
                    column=self.target_col,
                    message=f"Target column '{self.target_col}' is completely null (no valid labels).",
                    recommendation="Select a valid target column with labeled ground-truth values."
                ))
            elif target_analysis.null_count > 0:
                issues.append(AuditIssue(
                    severity="High",
                    category="Target",
                    column=self.target_col,
                    message=f"Target column '{self.target_col}' has {target_analysis.null_count} missing values ({target_analysis.null_pct}%).",
                    recommendation="Filter out unlabelled rows before model training (unsupervised training can keep them)."
                ))

            if target_analysis.is_constant:
                issues.append(AuditIssue(
                    severity="Critical",
                    category="Target",
                    column=self.target_col,
                    message=f"Target column '{self.target_col}' has only 1 unique value. Supervised learning cannot learn a boundary.",
                    recommendation="Choose a target with at least 2 distinct classes (classification) or variance (regression)."
                ))

            if target_analysis.infinite_count > 0:
                issues.append(AuditIssue(
                    severity="High",
                    category="Target",
                    column=self.target_col,
                    message=f"Target column '{self.target_col}' contains {target_analysis.infinite_count} infinite values.",
                    recommendation="Clean or filter out rows with infinite target values."
                ))

            # Classification specific target findings
            if target_analysis.task_type == "classification":
                if target_analysis.is_severely_imbalanced:
                    issues.append(AuditIssue(
                        severity="High",
                        category="Target",
                        column=self.target_col,
                        message=f"Severe class imbalance: minority class '{target_analysis.minority_class}' is only {target_analysis.minority_pct}% (imbalance ratio {target_analysis.imbalance_ratio}:1).",
                        recommendation="Use Stratified K-Fold, class-weighted loss (balanced), SMOTE resampling, and evaluate with PR-AUC / F1 instead of Accuracy."
                    ))
                elif target_analysis.is_moderately_imbalanced:
                    issues.append(AuditIssue(
                        severity="Medium",
                        category="Target",
                        column=self.target_col,
                        message=f"Moderate class imbalance: minority class '{target_analysis.minority_class}' represents {target_analysis.minority_pct}% of samples.",
                        recommendation="Monitor balanced accuracy or macro F1 score. Consider stratified splitting."
                    ))

            # Data Leakage suspects
            for suspect in target_analysis.leakage_suspects:
                issues.append(AuditIssue(
                    severity="Critical",
                    category="Data Leakage",
                    column=suspect["column"],
                    message=f"Potential Data Leakage in '{suspect['column']}': {suspect['reason']}",
                    recommendation=f"Drop '{suspect['column']}' from feature matrix. High correlation/equivalence indicates target proxy or future data."
                ))

        # -------------------------------------------------------------
        # 8. ML READINESS SCORE METHODOLOGY (Deterministic 0-100)
        # -------------------------------------------------------------
        # Deductions are transparently tracked
        deductions: List[ScoreDeduction] = []

        # Missing data penalty
        if missing_pct >= 50.0:
            deductions.append(ScoreDeduction("Missing Data", 25.0, f"Critical missingness ({missing_pct}% null cells)"))
        elif missing_pct >= 20.0:
            deductions.append(ScoreDeduction("Missing Data", 15.0, f"High missingness ({missing_pct}% null cells)"))
        elif missing_pct >= 5.0:
            deductions.append(ScoreDeduction("Missing Data", 8.0, f"Moderate missingness ({missing_pct}% null cells)"))
        elif missing_pct > 0.0:
            deductions.append(ScoreDeduction("Missing Data", 3.0, f"Minor missingness ({missing_pct}% null cells)"))

        # Individual empty columns
        n_empty = len(classified_cols["empty"])
        if n_empty > 0:
            p = min(12.0, n_empty * 4.0)
            deductions.append(ScoreDeduction("Empty Columns", p, f"{n_empty} completely empty column(s)"))

        # Duplicate rows penalty
        if dup_pct >= 25.0:
            deductions.append(ScoreDeduction("Duplicates", 10.0, f"High row duplication ({dup_pct}%)"))
        elif dup_pct >= 5.0:
            deductions.append(ScoreDeduction("Duplicates", 6.0, f"Moderate row duplication ({dup_pct}%)"))
        elif dup_pct > 0.0:
            deductions.append(ScoreDeduction("Duplicates", 2.0, f"Minor row duplication ({dup_pct}%)"))

        # Infinite values
        n_inf_cols = len(classified_cols["infinite"])
        if n_inf_cols > 0:
            deductions.append(ScoreDeduction("Infinite Values", 10.0, f"{n_inf_cols} column(s) contain infinite numbers (+/- inf)"))

        # Constant columns
        n_const = len(classified_cols["constant"])
        if n_const > 0:
            p = min(10.0, n_const * 3.0)
            deductions.append(ScoreDeduction("Constant Columns", p, f"{n_const} zero-variance column(s)"))

        # Near constant columns
        n_near_const = len(classified_cols["near_constant"])
        if n_near_const > 0:
            p = min(6.0, n_near_const * 2.0)
            deductions.append(ScoreDeduction("Near-Constant Columns", p, f"{n_near_const} column(s) with >=95% single value"))

        # ID-like columns
        n_ids = len(classified_cols["id_like"])
        if n_ids > 0:
            p = min(8.0, n_ids * 2.5)
            deductions.append(ScoreDeduction("ID-like Columns", p, f"{n_ids} potential identifier column(s) detected"))

        # Small sample size
        if n_rows < 20:
            deductions.append(ScoreDeduction("Dataset Size", 15.0, f"Extremely small sample size ({n_rows} rows)"))
        elif n_rows < 50:
            deductions.append(ScoreDeduction("Dataset Size", 7.0, f"Small sample size ({n_rows} rows)"))

        # High dimensionality (cols > rows)
        if n_cols > n_rows and n_rows >= 10:
            deductions.append(ScoreDeduction("Dimensionality", 5.0, f"Feature count exceeds row count ({n_cols} cols > {n_rows} rows)"))

        # Target specific penalties
        if target_analysis:
            if target_analysis.valid_count == 0:
                deductions.append(ScoreDeduction("Target Validity", 40.0, "Target column contains no valid ground-truth values"))
            elif target_analysis.is_constant:
                deductions.append(ScoreDeduction("Target Validity", 30.0, "Target column has only 1 unique value"))
            elif target_analysis.null_pct > 0:
                p = min(15.0, round(target_analysis.null_pct * 0.3, 1))
                deductions.append(ScoreDeduction("Target Missingness", p, f"Target has {target_analysis.null_pct}% missing labels"))

            if target_analysis.is_severely_imbalanced:
                deductions.append(ScoreDeduction("Class Imbalance", 12.0, f"Severe class imbalance (ratio {target_analysis.imbalance_ratio}:1)"))
            elif target_analysis.is_moderately_imbalanced:
                deductions.append(ScoreDeduction("Class Imbalance", 5.0, f"Moderate class imbalance (ratio {target_analysis.imbalance_ratio}:1)"))

            if len(target_analysis.leakage_suspects) > 0:
                deductions.append(ScoreDeduction("Data Leakage", 25.0, f"Detected {len(target_analysis.leakage_suspects)} potential target leakage feature(s)"))

        # Calculate final score
        total_penalty = sum(d.penalty for d in deductions)
        raw_score = 100.0 - total_penalty
        final_score = int(max(0, min(100, round(raw_score))))

        # If any Critical issue exists, cap score at 40
        has_critical = any(issue.severity == "Critical" for issue in issues)
        if has_critical and final_score > 40:
            deductions.append(ScoreDeduction("Critical Override", final_score - 40, "Score capped at 40 due to presence of Critical blockers."))
            final_score = 40

        # Assign Grade
        if final_score >= 85:
            grade = "Excellent (ML Ready)"
        elif final_score >= 70:
            grade = "Good (Minor issues)"
        elif final_score >= 50:
            grade = "Fair (Needs cleaning)"
        elif final_score >= 25:
            grade = "Poor (Significant cleaning required)"
        else:
            grade = "Critical (Unusable as-is)"

        # Overall severity
        if has_critical:
            overall_sev = "Critical"
        elif any(issue.severity == "High" for issue in issues):
            overall_sev = "High"
        elif any(issue.severity == "Medium" for issue in issues):
            overall_sev = "Medium"
        elif any(issue.severity == "Low" for issue in issues):
            overall_sev = "Low"
        else:
            overall_sev = "Info"

        # High-level warnings and recommendations
        for issue in issues:
            if issue.severity in ("Critical", "High"):
                warnings.append(f"[{issue.severity.upper()}] {issue.message}")
            if issue.recommendation and issue.recommendation not in recommendations:
                recommendations.append(issue.recommendation)

        if not recommendations:
            recommendations.append("Dataset passed quality checks and is ready for ML modeling.")

        outlier_summary = {
            "total_outliers": total_outlier_count,
            "affected_columns": classified_cols["outliers"],
            "columns_with_outliers_count": len(classified_cols["outliers"]),
            "details": {
                c: {
                    "count": col_profiles[c].outlier_count,
                    "percentage": col_profiles[c].outlier_pct,
                    "lower_bound": col_profiles[c].outlier_lower_bound,
                    "upper_bound": col_profiles[c].outlier_upper_bound
                }
                for c in classified_cols["outliers"]
            }
        }

        dataset_stats = {
            "rows": n_rows,
            "columns": n_cols,
            "total_cells": total_cells,
            "missing_cells": total_missing,
            "missing_pct": missing_pct,
            "duplicate_rows": dup_rows,
            "duplicate_pct": dup_pct,
            "numeric_cols_count": len(classified_cols["numeric"]),
            "categorical_cols_count": len(classified_cols["categorical"]),
            "datetime_cols_count": len(classified_cols["datetime"]),
            "boolean_cols_count": len(classified_cols["boolean"]),
            "unusable_cols_count": len(classified_cols["unusable"])
        }

        return AuditResult(
            score=final_score,
            grade=grade,
            severity=overall_sev,
            dataset_stats=dataset_stats,
            detected_issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            column_profiles=col_profiles,
            classified_columns=classified_cols,
            outlier_summary=outlier_summary,
            target_analysis=target_analysis,
            score_breakdown=deductions
        )


# =====================================================================
# CONVENIENCE API
# =====================================================================

def run_data_audit(df: pd.DataFrame, target_col: Optional[str] = None) -> AuditResult:
    """Convenience function to audit a dataframe."""
    auditor = DataAuditor(df, target_col=target_col)
    return auditor.audit()


# =====================================================================
# STREAMLIT UI DASHBOARD RENDERING HELPERS
# =====================================================================

def render_audit_summary_cards(audit: AuditResult):
    """
    Renders a compact, high-impact Dataset Health Overview suitable for
    the Upload step (Step 1) or top-of-page status banners.
    """
    import streamlit as st

    # Color tokens for score
    if audit.score >= 85:
        color = "#10b981"
        badge_class = "badge-success"
    elif audit.score >= 70:
        color = "#77ADBF"
        badge_class = "badge-primary"
    elif audit.score >= 50:
        color = "#f59e0b"
        badge_class = "badge-warning"
    else:
        color = "#f43f5e"
        badge_class = "badge-error"

    stats = audit.dataset_stats
    issues_count = len(audit.detected_issues)
    critical_or_high = [i for i in audit.detected_issues if i.severity in ("Critical", "High")]

    st.markdown(f"""
    <div class="glass-panel" style="border-left:4px solid {color};padding:1.2rem 1.5rem;margin:1rem 0;">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
            <div>
                <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;font-weight:700">
                    DIAGNOSTIC AUDIT
                </div>
                <div style="font-size:1.35rem;font-weight:700;color:var(--text);margin-top:0.2rem;display:flex;align-items:baseline;gap:0.5rem">
                    ML Readiness Score: <span style="color:{color};font-family:'Geist Mono',monospace;font-weight:600;font-size:1.6rem">{audit.score}</span><span style="font-size:0.9rem;color:var(--muted)">/100</span>
                    <span class="badge {badge_class}">{audit.grade}</span>
                </div>
                <div style="font-size:0.78rem;color:var(--text-secondary);margin-top:0.3rem">
                    Application-level diagnostic score based on measurable dataset characteristics.
                </div>
            </div>
            <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
                <span class="badge badge-primary">{stats.get('numeric_cols_count', 0)} Numeric</span>
                <span class="badge badge-success">{stats.get('categorical_cols_count', 0)} Categorical</span>
                {f'<span class="badge badge-warning">{stats["datetime_cols_count"]} Datetime</span>' if stats.get('datetime_cols_count', 0) > 0 else ''}
                <span class="badge {'badge-error' if critical_or_high else 'badge-primary'}">
                    {issues_count} Potential Issue{'s' if issues_count != 1 else ''}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4-card metric overview
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{stats["rows"]:,}</div><div class="metric-label">Total Rows</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{stats["columns"]:,}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{stats["missing_pct"]}%</div><div class="metric-label">Missing Cells</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{stats["duplicate_pct"]}%</div><div class="metric-label">Duplicate Rows</div></div>', unsafe_allow_html=True)

    # If critical / high issues exist, show an informative alert
    if critical_or_high:
        st.markdown(f"""
        <div class="warn-card" style="margin-top:0.8rem">
            <strong>High Priority Findings ({len(critical_or_high)}):</strong><br>
            <ul style="margin:0.4rem 0 0 1.2rem;padding:0;font-size:0.85rem">
                {''.join(f'<li><strong>[{i.severity}]</strong> {i.message}</li>' for i in critical_or_high[:3])}
            </ul>
            <div style="font-size:0.75rem;color:var(--muted);margin-top:0.4rem">
                Full diagnostics and recommendations available in <strong>Step 2 (Data Profiling → ML Readiness Audit)</strong>.
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_audit_full_dashboard(audit: AuditResult, key_suffix: str = ""):
    """
    Renders the comprehensive Data Quality & ML Readiness Audit tab inside Step 2 (Data Profiling).
    """
    import streamlit as st

    # Score color
    if audit.score >= 85:
        score_color = "#10b981"
        badge_cls = "badge-success"
    elif audit.score >= 70:
        score_color = "#77ADBF"
        badge_cls = "badge-primary"
    elif audit.score >= 50:
        score_color = "#f59e0b"
        badge_cls = "badge-warning"
    else:
        score_color = "#f43f5e"
        badge_cls = "badge-error"

    stats = audit.dataset_stats

    # Header Card
    st.markdown(f"""
    <div class="glass-panel" style="border-left:4px solid {score_color};padding:1.5rem 1.8rem;margin-bottom:1.5rem">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1.5rem">
            <div>
                <div style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;font-weight:700">
                    ML READINESS DIAGNOSTIC
                </div>
                <div style="display:flex;align-items:baseline;gap:0.4rem;margin-top:0.3rem">
                    <span style="font-size:2.8rem;font-weight:700;color:{score_color};font-family:'Geist Mono',monospace;line-height:1">{audit.score}</span>
                    <span style="font-size:1.3rem;color:var(--muted);font-weight:600">/ 100</span>
                    <span class="badge {badge_cls}" style="margin-left:0.5rem;font-size:0.8rem;padding:0.25rem 0.65rem">{audit.grade}</span>
                </div>
                <div style="color:var(--text-secondary);font-size:0.85rem;margin-top:0.4rem">
                    Deterministic 100-point data-quality assessment for automated machine learning workflows.
                </div>
            </div>
            <div style="text-align:right">
                <div style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:0.3rem">Overall Risk Level</div>
                <span class="badge {'badge-error' if audit.severity in ('Critical', 'High') else 'badge-success'}" style="font-size:0.85rem;padding:0.3rem 0.8rem">
                    {audit.severity.upper()} SEVERITY
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(f'<div class="metric-card"><div class="metric-value">{stats["rows"]:,}</div><div class="metric-label">Rows</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-value">{stats["columns"]:,}</div><div class="metric-label">Columns</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-value">{stats["missing_pct"]}%</div><div class="metric-label">Missing Data</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-value">{stats["duplicate_rows"]:,}</div><div class="metric-label">Duplicates</div></div>', unsafe_allow_html=True)
    m5.markdown(f'<div class="metric-card"><div class="metric-value">{len(audit.detected_issues)}</div><div class="metric-label">Findings</div></div>', unsafe_allow_html=True)

    # Transparent Score Methodology Expander
    with st.expander("ML Readiness Scoring Methodology & Deduction Breakdown", expanded=False):
        st.markdown("""
        **Scoring Methodology (0–100 Bounded Scale):**
        - Every dataset starts at **100 base points**.
        - Deductions are deterministically computed based on measurable risks across missing data severity, row duplicates, constant/near-constant columns, statistical outliers, infinite values, high cardinality, and target distribution.
        - The score is strictly bounded $[0, 100]$ and represents a practical diagnostic evaluation for supervised and tabular ML workflows.
        """)
        if audit.score_breakdown:
            breakdown_rows = []
            for item in audit.score_breakdown:
                breakdown_rows.append({
                    "Category": item.category,
                    "Penalty": f"-{item.penalty:.1f} pts",
                    "Reason": item.reason
                })
            b_df = pd.DataFrame(breakdown_rows)
            st.dataframe(b_df, use_container_width=True)
            st.caption(f"**Total Penalties Applied:** -{sum(d.penalty for d in audit.score_breakdown):.1f} points ➔ Final Score: **{audit.score}/100**")
        else:
            st.success("No deductions applied! Perfect 100/100 readiness score.")

    # Actionable Recommendations Box
    if audit.recommendations:
        st.markdown("""
        <div class="ai-card" style="margin:1rem 0">
            <strong>Actionable Next Steps:</strong>
            <ul style="margin:0.5rem 0 0 1.2rem;padding:0;font-size:0.85rem">
        """ + "".join(f"<li style='margin-bottom:0.3rem'>{r}</li>" for r in audit.recommendations) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Sub-tabs for Audit Details
    tab_issues, tab_cols, tab_outliers, tab_target = st.tabs([
        f"Issues & Findings ({len(audit.detected_issues)})",
        "Column Intelligence",
        "Outlier Diagnostics",
        "Target & Leakage Audit"
    ])

    # 1. ISSUES TAB
    with tab_issues:
        if not audit.detected_issues:
            st.success("No quality issues detected in this dataset! You are ready to proceed.")
        else:
            sev_filter = st.selectbox(
                "Filter by severity",
                ["All", "Critical", "High", "Medium", "Low", "Info"],
                key=f"sev_filter_{key_suffix}"
            )
            filtered_issues = audit.detected_issues if sev_filter == "All" else [i for i in audit.detected_issues if i.severity == sev_filter]

            st.markdown(f"Displaying **{len(filtered_issues)}** finding(s):")
            for idx, issue in enumerate(filtered_issues):
                s = issue.severity
                badge_type = "badge-error" if s in ("Critical", "High") else ("badge-warning" if s == "Medium" else "badge-success")
                border_col = "#f43f5e" if s in ("Critical", "High") else ("#f59e0b" if s == "Medium" else "#10b981")
                col_text = f" &bull; <code>{issue.column}</code>" if issue.column else ""

                st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid {border_col};padding:1rem 1.25rem;margin-bottom:0.75rem">
                    <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem">
                        <span class="badge {badge_type}">{s.upper()}</span>
                        <span style="font-weight:700;font-size:0.88rem;color:var(--text)">{issue.category}</span>
                        <span style="color:var(--muted);font-size:0.8rem">{col_text}</span>
                    </div>
                    <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.45">
                        {issue.message}
                    </div>
                    <div style="font-size:0.8rem;color:var(--accent);margin-top:0.4rem">
                        <strong>Action:</strong> {issue.recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 2. COLUMN INTELLIGENCE TAB
    with tab_cols:
        st.markdown("#### Comprehensive Column Classification & Anomaly Tags")
        col_rows = []
        for cname, p in audit.column_profiles.items():
            flags = []
            if p.is_id_like:
                flags.append("ID-like")
            if p.is_high_cardinality:
                flags.append("High Cardinality")
            if p.is_constant:
                flags.append("Constant")
            if p.is_near_constant:
                flags.append("Near-Constant")
            if p.is_empty:
                flags.append("Empty")
            elif p.null_pct >= 50:
                flags.append("High Missing")
            if p.infinite_count > 0:
                flags.append("Infinite")
            if p.outlier_count > 0:
                flags.append(f"Outliers ({p.outlier_pct}%)")

            flag_str = ", ".join(flags) if flags else "Normal"
            col_rows.append({
                "Column": cname,
                "Detected Type": p.detected_type,
                "Dtype": p.dtype,
                "Missing %": f"{p.null_pct}%",
                "Unique Count": p.unique_count,
                "Unique Ratio": f"{p.unique_ratio*100:.1f}%",
                "Quality Flags": flag_str,
                "Dominant / Top Value": str(p.top_value) if p.top_value is not None else "N/A"
            })

        col_df = pd.DataFrame(col_rows)
        st.dataframe(col_df, use_container_width=True, height=min(400, len(col_df) * 35 + 40))

    # 3. OUTLIER DIAGNOSTICS TAB
    with tab_outliers:
        st.markdown("#### Outlier Analysis (Interquartile Range Method)")
        out_summary = audit.outlier_summary
        total_outs = out_summary.get("total_outliers", 0)
        aff_cols = out_summary.get("affected_columns", [])

        if total_outs == 0:
            st.success("No statistical outliers detected in any numeric columns.")
        else:
            st.info(f"Found **{total_outs:,}** total statistical outliers across **{len(aff_cols)}** numeric column(s) using the $[Q_1 - 1.5\\times IQR, Q_3 + 1.5\\times IQR]$ rule.")
            out_rows = []
            for col_name in aff_cols:
                info = out_summary["details"][col_name]
                out_rows.append({
                    "Column": col_name,
                    "Outlier Count": info["count"],
                    "Outlier %": f"{info['percentage']}%",
                    "Lower Bound (Q1 - 1.5*IQR)": f"{info['lower_bound']:.4f}" if info['lower_bound'] is not None else "N/A",
                    "Upper Bound (Q3 + 1.5*IQR)": f"{info['upper_bound']:.4f}" if info['upper_bound'] is not None else "N/A"
                })
            st.dataframe(pd.DataFrame(out_rows), use_container_width=True)

    # 4. TARGET & LEAKAGE AUDIT TAB
    with tab_target:
        ta = audit.target_analysis
        if ta is None:
            st.info("No target column was specified for this audit. Target diagnostics run automatically in **Step 7 (Model Training)** when you select what to predict.")
        else:
            st.markdown(f"#### Target Column Audit: `{ta.target_col}`")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.markdown(f"""
                <div class="ai-card">
                    <strong>Task Type:</strong> <span class="badge badge-purple">{ta.task_type.upper()}</span><br><br>
                    <strong>Valid Values:</strong> {ta.valid_count:,} / {ta.total_count:,} ({100-ta.null_pct:.1f}%)<br>
                    <strong>Missing Values:</strong> {ta.null_count:,} ({ta.null_pct}%)<br>
                    <strong>Infinite Values:</strong> {ta.infinite_count:,}<br>
                    <strong>Constant Target:</strong> {"Yes (CRITICAL)" if ta.is_constant else "No (Valid variance)"}
                </div>
                """, unsafe_allow_html=True)
            with t_col2:
                if ta.task_type == "classification" and ta.class_distribution:
                    imb_badge = "badge-red" if ta.is_severely_imbalanced else "badge-yellow" if ta.is_moderately_imbalanced else "badge-green"
                    st.markdown(f"""
                    <div class="ai-card">
                        <strong>Classes:</strong> {ta.n_classes}<br><br>
                        <strong>Balance Status:</strong> <span class="badge {imb_badge}">
                            {"Severely Imbalanced" if ta.is_severely_imbalanced else "Moderately Imbalanced" if ta.is_moderately_imbalanced else "Balanced"}
                        </span><br>
                        <strong>Minority Class:</strong> '{ta.minority_class}' ({ta.minority_pct}%)<br>
                        <strong>Imbalance Ratio:</strong> {ta.imbalance_ratio}:1
                    </div>
                    """, unsafe_allow_html=True)
                elif ta.task_type == "regression" and ta.regression_stats:
                    rs = ta.regression_stats
                    st.markdown(f"""
                    <div class="ai-card">
                        <strong>Target Distribution:</strong><br><br>
                        &bull; <strong>Mean:</strong> {rs.get('mean')}<br>
                        &bull; <strong>Std Dev:</strong> {rs.get('std')}<br>
                        &bull; <strong>Range:</strong> [{rs.get('min')} ➔ {rs.get('max')}]<br>
                        &bull; <strong>Skewness:</strong> {rs.get('skewness')}
                    </div>
                    """, unsafe_allow_html=True)

            # Data leakage alerts
            if ta.leakage_suspects:
                st.markdown("#### Data Leakage Suspects Detected")
                for s in ta.leakage_suspects:
                    st.markdown(f"""
                    <div class="warn-card" style="border-left:4px solid #ff6b6b">
                        <strong>CRITICAL LEAKAGE ALERT:</strong> Column <code>{s['column']}</code> has extreme correlation or exact match ({s['reason']}).<br>
                        <em>Recommendation: Drop <code>{s['column']}</code> from features before training to prevent false 100% test accuracy.</em>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No target leakage features detected.")


def render_target_preflight_check(audit: AuditResult):
    """
    Renders a targeted pre-flight checklist alert before running Model Training in Step 7.
    """
    import streamlit as st

    ta = audit.target_analysis
    if not ta:
        return

    crit_or_high = [i for i in audit.detected_issues if i.category in ("Target", "Data Leakage") and i.severity in ("Critical", "High")]
    
    if crit_or_high:
        st.markdown("""
        <div class="warn-card" style="border-left:4px solid #ff6b6b;margin-bottom:1rem">
            <span style="font-size:1.1rem;font-weight:800;color:#ff6b6b">Pre-Flight Target & Quality Alerts</span><br>
            <span style="font-size:0.85rem;color:var(--text)">The audit engine identified potential issues with the selected target or features:</span>
            <ul style="margin:0.4rem 0 0 1.2rem;padding:0;font-size:0.82rem">
        """ + "".join(f"<li><strong>[{issue.severity}]</strong> {issue.message} ➔ <em>{issue.recommendation}</em></li>" for issue in crit_or_high) + """
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif ta.is_moderately_imbalanced:
        st.markdown(f"""
        <div class="ai-card" style="margin-bottom:1rem;font-size:0.85rem">
            <strong>Note on Class Balance:</strong> Minority class '{ta.minority_class}' represents {ta.minority_pct}% of instances. Stratified cross-validation is recommended.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(0,212,170,0.08);border:1px solid rgba(0,212,170,0.3);border-radius:8px;padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#00d4aa">
            <strong>Target Quality Passed:</strong> No leakage, valid variance, and healthy class distribution. Ready for model training.
        </div>
        """, unsafe_allow_html=True)

