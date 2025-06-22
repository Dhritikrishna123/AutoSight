import pandas as pd
import numpy as np
import re
from typing import Dict, List, Any

def get_cleaning_suggestions(df: pd.DataFrame) -> dict:
    """
    Enhanced data quality detection system that identifies various data issues
    and provides comprehensive cleaning suggestions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to analyze
        
    Returns:
    --------
    dict
        Dictionary containing cleaning suggestions with issue types as keys
        and affected columns/details as values
    """
    suggestions = {}
    
    if df.empty:
        return suggestions
    
    # 1. Drop columns with >50% missing values
    high_missing = df.columns[df.isnull().mean() > 0.5].tolist()
    if high_missing:
        suggestions["Drop columns with >50% missing"] = high_missing
    
    # 2. Detect columns with moderate missing values (20-50%)
    moderate_missing = df.columns[(df.isnull().mean() > 0.2) & (df.isnull().mean() <= 0.5)].tolist()
    if moderate_missing:
        suggestions["Consider imputation for moderate missing values"] = moderate_missing
    
    # 3. Enhanced string normalization detection
    messy_strings = []
    for col in df.select_dtypes(include="object"):
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().astype(str).unique()
            if len(unique_vals) > 0:
                # Check for various string inconsistencies
                has_whitespace_issues = any(str(u).strip() != str(u) for u in unique_vals)
                has_case_issues = len(set(str(u).lower() for u in unique_vals)) < len(unique_vals)
                has_punctuation = any("," in str(u) or ";" in str(u) or ":" in str(u) for u in unique_vals)
                has_special_chars = any(bool(re.search(r'[^\w\s-]', str(u))) for u in unique_vals[:50])  # Check first 50 for performance
                
                if has_whitespace_issues or has_case_issues or has_punctuation or has_special_chars:
                    messy_strings.append(col)
    
    if messy_strings:
        suggestions["Normalize inconsistent strings"] = messy_strings
    
    # 4. Detect columns that could be converted to numeric
    potential_numeric = []
    for col in df.select_dtypes(include="object"):
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().astype(str).head(100)  # Sample for performance
            if len(sample_vals) > 0:
                # Check for numeric patterns
                numeric_pattern_count = sum(1 for val in sample_vals 
                                          if re.search(r'^\s*[-+]?\d*\.?\d+([eE][-+]?\d+)?\s*$', str(val)) or
                                             re.search(r'^\s*[$€£¥]?\s*[-+]?\d{1,3}(,\d{3})*\.?\d*\s*%?\s*$', str(val)))
                
                if numeric_pattern_count / len(sample_vals) > 0.7:  # 70% appear to be numeric
                    potential_numeric.append(col)
    
    if potential_numeric:
        suggestions["Convert text columns to numeric"] = potential_numeric
    
    # 5. Detect potential datetime columns
    potential_datetime = []
    for col in df.select_dtypes(include="object"):
        if df[col].dtype == 'object':
            sample_vals = df[col].dropna().astype(str).head(50)
            if len(sample_vals) > 0:
                datetime_patterns = [
                    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
                    r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or MM/DD/YYYY
                    r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}',  # YYYY-MM-DD HH:MM
                    r'\w{3}\s\d{1,2},?\s\d{4}',  # Mon DD, YYYY
                ]
                
                datetime_count = sum(1 for val in sample_vals
                                   if any(re.search(pattern, str(val)) for pattern in datetime_patterns))
                
                if datetime_count / len(sample_vals) > 0.6:  # 60% appear to be dates
                    potential_datetime.append(col)
    
    if potential_datetime:
        suggestions["Convert to datetime format"] = potential_datetime
    
    # 6. Enhanced outlier detection with multiple methods
    outliers = {}
    for col in df.select_dtypes(include="number"):
        if df[col].notna().sum() > 0:  # Only process if there are non-null values
            # IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            iqr_outliers = df[(df[col] < lower) | (df[col] > upper)]
            
            # Z-score method (for additional validation)
            try:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                zscore_outliers = df[z_scores > 3]
                
                # Use the more conservative count
                outlier_count = min(len(iqr_outliers), len(zscore_outliers))
                if outlier_count > 0:
                    outliers[col] = outlier_count
            except (ZeroDivisionError, ValueError):
                # Fallback to IQR method only
                if len(iqr_outliers) > 0:
                    outliers[col] = len(iqr_outliers)
    
    if outliers:
        suggestions["Potential outliers"] = outliers
    
    # 7. Detect duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        suggestions["Remove duplicate rows"] = f"{duplicate_count} duplicate rows found"
    
    # 8. Detect columns with single unique value (constant columns)
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        suggestions["Drop constant/single-value columns"] = constant_cols
    
    # 9. Detect mixed data types within columns
    mixed_type_cols = []
    for col in df.select_dtypes(include="object"):
        if df[col].notna().sum() > 0:
            sample_vals = df[col].dropna().head(100)
            type_set = set(type(val).__name__ for val in sample_vals)
            if len(type_set) > 1:
                mixed_type_cols.append(col)
    
    if mixed_type_cols:
        suggestions["Fix mixed data types"] = mixed_type_cols
    
    # 10. Detect columns suitable for categorical conversion
    potential_categorical = []
    for col in df.select_dtypes(include="object"):
        if df[col].notna().sum() > 0:
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            # Good candidates: low unique values, high repetition
            if (unique_count <= 20 and 
                unique_count > 1 and 
                total_count > 0 and 
                unique_count / total_count < 0.1):  # Less than 10% unique values
                potential_categorical.append(col)
    
    if potential_categorical:
        suggestions["Convert to categorical for memory efficiency"] = potential_categorical
    
    # 11. Detect extremely long text values that might need truncation
    long_text_cols = []
    for col in df.select_dtypes(include="object"):
        if df[col].notna().sum() > 0:
            max_length = df[col].astype(str).str.len().max()
            if max_length > 1000:  # Arbitrary threshold for "too long"
                long_text_cols.append(col)
    
    if long_text_cols:
        suggestions["Consider truncating extremely long text"] = long_text_cols
    
    # 12. Detect columns with leading/trailing whitespace
    whitespace_cols = []
    for col in df.select_dtypes(include="object"):
        if df[col].notna().sum() > 0:
            sample_vals = df[col].dropna().astype(str).head(100)
            if any(val != val.strip() for val in sample_vals):
                whitespace_cols.append(col)
    
    if whitespace_cols:
        suggestions["Remove leading/trailing whitespace"] = whitespace_cols
    
    # 13. Detect columns with encoding issues
    encoding_issue_cols = []
    for col in df.select_dtypes(include="object"):
        if df[col].notna().sum() > 0:
            sample_vals = df[col].dropna().astype(str).head(50)
            # Look for common encoding artifacts
            encoding_patterns = [r'â€™', r'â€œ', r'â€', r'Ã', r'\\x', r'\\u']
            has_encoding_issues = any(
                any(re.search(pattern, str(val)) for pattern in encoding_patterns)
                for val in sample_vals
            )
            if has_encoding_issues:
                encoding_issue_cols.append(col)
    
    if encoding_issue_cols:
        suggestions["Fix text encoding issues"] = encoding_issue_cols
    
    # 14. Detect negative values in columns that should be positive
    negative_value_cols = []
    for col in df.select_dtypes(include="number"):
        # Heuristic: if column name suggests it should be positive
        if any(keyword in col.lower() for keyword in ['age', 'count', 'quantity', 'amount', 'price', 'cost', 'distance', 'length', 'width', 'height', 'weight', 'size']):
            if (df[col] < 0).any():
                negative_value_cols.append(col)
    
    if negative_value_cols:
        suggestions["Review negative values in positive-expected columns"] = negative_value_cols
    
    # 15. Memory optimization suggestions
    memory_optimization_cols = []
    
    # Check for over-sized integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_max = df[col].max()
        col_min = df[col].min()
        
        if col_min >= 0 and col_max <= 255:
            memory_optimization_cols.append(f"{col} (int64 -> uint8)")
        elif col_min >= -128 and col_max <= 127:
            memory_optimization_cols.append(f"{col} (int64 -> int8)")
    
    # Check for over-sized float columns
    for col in df.select_dtypes(include=['float64']).columns:
        if df[col].notna().sum() > 0:
            col_max = df[col].max()
            col_min = df[col].min()
            
            if (col_min >= np.finfo(np.float32).min and 
                col_max <= np.finfo(np.float32).max):
                memory_optimization_cols.append(f"{col} (float64 -> float32)")
    
    if memory_optimization_cols:
        suggestions["Optimize data types for memory efficiency"] = memory_optimization_cols
    
    return suggestions