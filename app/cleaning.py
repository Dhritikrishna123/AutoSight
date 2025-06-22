import pandas as pd
import numpy as np
import re
from typing import Union, List, Optional, Dict, Any
import warnings

def deep_clean_dataframe(
    df: pd.DataFrame, 
    missing_threshold: float = 0.5,
    numeric_conversion_threshold: float = 0.9,
    custom_na_values: Optional[List[str]] = None,
    preserve_columns: Optional[List[str]] = None,
    datetime_columns: Optional[List[str]] = None,
    categorical_threshold: int = 10,
    remove_duplicates: bool = True,
    standardize_text: bool = True,
    handle_outliers: bool = False,
    outlier_method: str = 'iqr',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Comprehensively clean a pandas DataFrame with robust error handling and enhanced features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to clean
    missing_threshold : float, default 0.5
        Drop columns with missing values above this threshold (0-1)
    numeric_conversion_threshold : float, default 0.9
        Only convert to numeric if success rate is above this threshold (0-1)
    custom_na_values : List[str], optional
        Additional values to treat as NaN
    preserve_columns : List[str], optional
        Column names to never drop (even if they exceed missing threshold)
    datetime_columns : List[str], optional
        Column names to attempt datetime conversion
    categorical_threshold : int, default 10
        Convert string columns to categorical if unique values <= threshold
    remove_duplicates : bool, default True
        Whether to remove duplicate rows
    standardize_text : bool, default True
        Whether to strip and lowercase text columns
    handle_outliers : bool, default False
        Whether to handle outliers in numeric columns
    outlier_method : str, default 'iqr'
        Method for outlier detection ('iqr', 'zscore', 'isolation')
    verbose : bool, default False
        Whether to print cleaning progress
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame
    """
    
    if df.empty:
        warnings.warn("Input DataFrame is empty")
        return df.copy()
    
    df_clean = df.copy()
    original_shape = df_clean.shape
    original_memory = df_clean.memory_usage(deep=True).sum()
    
    if verbose:
        print(f"Starting cleanup of DataFrame with shape: {original_shape}")
        print(f"Original memory usage: {original_memory / 1024**2:.2f} MB")
    
    # 1. Enhanced missing values handling with comprehensive NA detection
    default_na_values = [
        "?", "", "none", "None", "NULL", "null", "NaN", "nan", "NA", "na", 
        "N/A", "n/a", "#N/A", "#NULL!", "#DIV/0!", "undefined", "Undefined",
        "UNDEFINED", "-", "--", "---", "null", "Null", "MISSING", "missing",
        "Unknown", "unknown", "UNKNOWN", " ", "  ", "\t", "\n", "NIL", "nil",
        "VOID", "void", "Empty", "empty", "EMPTY", "Not Available", "not available"
    ]
    
    na_values = default_na_values + (custom_na_values or [])
    
    try:
        # Replace various representations of missing data
        df_clean = df_clean.replace(na_values, pd.NA)
        
        # Handle numeric representations of missing data
        df_clean = df_clean.replace([np.inf, -np.inf, 999999, -999999, 9999, -9999, 99999], pd.NA)
        
        if verbose:
            print(f"Replaced {len(na_values)} types of missing value representations")
        
    except Exception as e:
        if verbose:
            print(f"Warning: Error in NA replacement: {e}")
    
    # 2. Remove constant/single-value columns (except preserved ones)
    preserve_columns = preserve_columns or []
    constant_cols = []
    for col in df_clean.columns:
        if col not in preserve_columns and df_clean[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        df_clean = df_clean.drop(columns=constant_cols)
        if verbose:
            print(f"Dropped {len(constant_cols)} constant/single-value columns: {constant_cols}")
    
    # 3. Remove columns with excessive missing values (with protection for specified columns)
    columns_to_check = [col for col in df_clean.columns if col not in preserve_columns]
    
    if columns_to_check:
        missing_ratios = df_clean[columns_to_check].isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index.tolist()
        
        if cols_to_drop:
            df_clean = df_clean.drop(columns=cols_to_drop)
            if verbose:
                print(f"Dropped {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values: {cols_to_drop}")
    
    # 4. Remove duplicate rows
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        if verbose and duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")
    
    # 5. Enhanced text standardization and encoding fix
    if standardize_text:
        text_columns = df_clean.select_dtypes(include=['object']).columns
        
        for col in text_columns:
            try:
                # Handle mixed types in column
                df_clean[col] = df_clean[col].astype(str)
                
                # Only process non-null values
                mask = df_clean[col].notna() & (df_clean[col] != 'nan')
                if mask.any():
                    # Fix encoding issues first
                    encoding_fixes = {
                        'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€˜': "'",
                        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
                        'Ã': 'Á', 'Ã‰': 'É', 'Ã': 'Í', 'Ã"': 'Ó', 'Ãš': 'Ú'
                    }
                    
                    for bad_encoding, fix in encoding_fixes.items():
                        df_clean.loc[mask, col] = df_clean.loc[mask, col].str.replace(bad_encoding, fix, regex=False)
                    
                    # Comprehensive text cleaning
                    df_clean.loc[mask, col] = (
                        df_clean.loc[mask, col]
                        .str.strip()  # Remove leading/trailing whitespace
                        .str.replace(r'\s+', ' ', regex=True)  # Normalize multiple spaces
                        .str.replace(r'[^\w\s\-.,!?;:()\[\]{}"\']', '', regex=True)  # Remove special chars
                        .str.lower()  # Convert to lowercase
                        .replace('nan', pd.NA)  # Convert string 'nan' back to actual NaN
                    )
                    
                    # Remove empty strings after cleaning
                    df_clean.loc[df_clean[col] == '', col] = pd.NA
                    
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not standardize column {col}: {e}")
    
    # 6. Auto-detect and convert datetime columns
    potential_datetime_cols = []
    text_columns = df_clean.select_dtypes(include=['object']).columns
    
    for col in text_columns:
        try:
            if df_clean[col].notna().sum() > 0:
                sample_vals = df_clean[col].dropna().astype(str).head(50)
                datetime_patterns = [
                    r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # YYYY-MM-DD or YYYY/MM/DD
                    r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',  # MM-DD-YYYY or MM/DD/YYYY
                    r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}',  # YYYY-MM-DD HH:MM
                    r'\w{3}\s\d{1,2},?\s\d{4}',  # Mon DD, YYYY
                ]
                
                datetime_count = sum(1 for val in sample_vals
                                   if any(re.search(pattern, str(val)) for pattern in datetime_patterns))
                
                if datetime_count / len(sample_vals) > 0.6:  # 60% appear to be dates
                    potential_datetime_cols.append(col)
        except:
            pass
    
    # Add user-specified datetime columns
    all_datetime_cols = list(set((datetime_columns or []) + potential_datetime_cols))
    
    # 7. Handle datetime columns
    for col in all_datetime_cols:
        if col in df_clean.columns:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce', infer_datetime_format=True)
                if verbose:
                    print(f"Converted column '{col}' to datetime")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not convert column {col} to datetime: {e}")
    
    # 8. Enhanced numeric conversion with improved pattern matching
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                cleaned_col = df_clean[col].astype(str)
                
                def enhanced_numeric_extraction(val):
                    """Extract numeric values with improved pattern matching"""
                    if pd.isna(val) or val in ['nan', 'none', '']:
                        return pd.NA
                    
                    try:
                        val = str(val).strip().lower()
                        
                        # Handle percentage values
                        if '%' in val:
                            nums = re.findall(r'-?\d+(?:\.\d+)?', val)
                            if nums:
                                return float(nums[0]) / 100
                        
                        # Handle currency values (improved)
                        if any(symbol in val for symbol in ['$', '€', '£', '¥', '₹', '¢']):
                            # Remove currency symbols, commas, and spaces
                            cleaned = re.sub(r'[^\d.-]', '', val.replace(',', ''))
                            if cleaned and cleaned not in ['-', '.']:
                                try:
                                    return float(cleaned)
                                except ValueError:
                                    pass
                        
                        # Handle scientific notation
                        scientific_match = re.search(r'-?\d+(?:\.\d+)?[eE][-+]?\d+', val)
                        if scientific_match:
                            return float(scientific_match.group())
                        
                        # Handle fractions
                        fraction_match = re.search(r'(\d+)\s*/\s*(\d+)', val)
                        if fraction_match:
                            return float(fraction_match.group(1)) / float(fraction_match.group(2))
                        
                        # Handle comma-separated numbers (thousands)
                        comma_num_match = re.search(r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?', val)
                        if comma_num_match:
                            return float(comma_num_match.group().replace(',', ''))
                        
                        # Handle ranges and multiple numbers
                        nums = re.findall(r'-?\d+(?:\.\d+)?', val)
                        if len(nums) == 0:
                            return pd.NA
                        elif len(nums) == 1:
                            return float(nums[0])
                        else:
                            # For ranges, take the average
                            return sum(float(x) for x in nums) / len(nums)
                            
                    except (ValueError, TypeError, ZeroDivisionError):
                        return pd.NA
                
                new_col = cleaned_col.map(enhanced_numeric_extraction)
                
                # Only convert if high success rate and contains meaningful numeric data
                success_rate = new_col.notna().sum() / len(new_col) if len(new_col) > 0 else 0
                unique_values = new_col.dropna().nunique()
                
                if (success_rate >= numeric_conversion_threshold and 
                    unique_values > 1 and  # Avoid converting columns with only one unique value
                    new_col.notna().sum() >= 5):  # Need at least 5 valid numeric values
                    
                    df_clean[col] = pd.to_numeric(new_col, errors='coerce')
                    if verbose:
                        print(f"Converted column '{col}' to numeric (success rate: {success_rate:.2%})")
                        
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not process column {col} for numeric conversion: {e}")
    
    # 9. Enhanced categorical conversion with better logic
    text_columns = df_clean.select_dtypes(include=['object']).columns
    for col in text_columns:
        try:
            unique_count = df_clean[col].nunique()
            total_count = len(df_clean[col].dropna())
            
            # Better criteria for categorical conversion
            if (unique_count <= categorical_threshold and 
                unique_count > 1 and 
                total_count > 0 and
                unique_count / total_count < 0.1 and  # Less than 10% unique values
                total_count >= 10):  # Need sufficient data points
                
                df_clean[col] = df_clean[col].astype('category')
                if verbose:
                    print(f"Converted column '{col}' to categorical ({unique_count} unique values)")
                    
        except Exception as e:
            if verbose:
                print(f"Warning: Could not convert column {col} to categorical: {e}")
    
    # 10. Enhanced outlier handling with multiple methods
    if handle_outliers:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                if df_clean[col].notna().sum() < 10:  # Skip if too few data points
                    continue
                    
                outliers_mask = pd.Series(False, index=df_clean.index)
                
                if outlier_method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:  # Avoid division by zero
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outliers_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                    
                elif outlier_method == 'zscore':
                    mean_val = df_clean[col].mean()
                    std_val = df_clean[col].std()
                    
                    if std_val > 0:  # Avoid division by zero
                        z_scores = np.abs((df_clean[col] - mean_val) / std_val)
                        outliers_mask = z_scores > 3
                
                elif outlier_method == 'modified_zscore':
                    median_val = df_clean[col].median()
                    mad = np.median(np.abs(df_clean[col] - median_val))
                    
                    if mad > 0:
                        modified_z_scores = 0.6745 * (df_clean[col] - median_val) / mad
                        outliers_mask = np.abs(modified_z_scores) > 3.5
                
                # Only remove outliers if they're not too many (avoid removing too much data)
                outlier_count = outliers_mask.sum()
                total_valid = df_clean[col].notna().sum()
                
                if outlier_count > 0 and outlier_count / total_valid < 0.1:  # Less than 10% outliers
                    df_clean.loc[outliers_mask, col] = pd.NA
                    if verbose:
                        print(f"Removed {outlier_count} outliers from column '{col}' using {outlier_method} method")
                        
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not handle outliers in column {col}: {e}")
    
    # 11. Handle negative values in columns that should be positive
    positive_keywords = ['age', 'count', 'quantity', 'amount', 'price', 'cost', 
                        'distance', 'length', 'width', 'height', 'weight', 'size',
                        'area', 'volume', 'speed', 'rate', 'percent', 'percentage']
    
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        try:
            if any(keyword in col.lower() for keyword in positive_keywords):
                negative_mask = df_clean[col] < 0
                if negative_mask.any():
                    # Replace with absolute value or NaN based on context
                    if 'percent' in col.lower() or 'rate' in col.lower():
                        df_clean.loc[negative_mask, col] = pd.NA  # Negative percentages might be invalid
                    else:
                        df_clean.loc[negative_mask, col] = df_clean.loc[negative_mask, col].abs()
                    
                    if verbose:
                        print(f"Handled {negative_mask.sum()} negative values in positive-expected column '{col}'")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not handle negative values in column {col}: {e}")
    
    # 12. Enhanced data type optimization for memory efficiency
    try:
        # Optimize integer columns
        int_cols = df_clean.select_dtypes(include=['int64', 'int32']).columns
        for col in int_cols:
            if df_clean[col].notna().sum() > 0:
                col_min = df_clean[col].min()
                col_max = df_clean[col].max()
                
                if col_min >= 0:  # Unsigned integers
                    if col_max <= 255:
                        df_clean[col] = df_clean[col].astype('uint8')
                    elif col_max <= 65535:
                        df_clean[col] = df_clean[col].astype('uint16')
                    elif col_max <= 4294967295:
                        df_clean[col] = df_clean[col].astype('uint32')
                else:  # Signed integers
                    if col_min >= -128 and col_max <= 127:
                        df_clean[col] = df_clean[col].astype('int8')
                    elif col_min >= -32768 and col_max <= 32767:
                        df_clean[col] = df_clean[col].astype('int16')
                    elif col_min >= -2147483648 and col_max <= 2147483647:
                        df_clean[col] = df_clean[col].astype('int32')
        
        # Optimize float columns
        float_cols = df_clean.select_dtypes(include=['float64']).columns
        for col in float_cols:
            if df_clean[col].notna().sum() > 0:
                col_min = df_clean[col].min()
                col_max = df_clean[col].max()
                
                # Check if values can fit in float32
                if (not np.isinf(col_min) and not np.isinf(col_max) and
                    col_min >= np.finfo(np.float32).min and 
                    col_max <= np.finfo(np.float32).max):
                    
                    # Additional check for precision loss
                    original_unique = df_clean[col].nunique()
                    test_conversion = df_clean[col].astype('float32')
                    new_unique = test_conversion.nunique()
                    
                    # Only convert if we don't lose too much precision
                    if new_unique >= original_unique * 0.95:  # Allow 5% precision loss
                        df_clean[col] = test_conversion
                
    except Exception as e:
        if verbose:
            print(f"Warning: Could not optimize data types: {e}")
    
    # 13. Final cleanup and validation
    try:
        # Remove any remaining empty string columns that weren't caught
        for col in df_clean.select_dtypes(include=['object']).columns:
            empty_mask = df_clean[col].isin(['', ' ', '  '])
            if empty_mask.any():
                df_clean.loc[empty_mask, col] = pd.NA
        
        # Remove columns that became entirely null after cleaning
        completely_null_cols = df_clean.columns[df_clean.isnull().all()].tolist()
        if completely_null_cols:
            df_clean = df_clean.drop(columns=completely_null_cols)
            if verbose:
                print(f"Dropped {len(completely_null_cols)} columns that became entirely null after cleaning")
    
    except Exception as e:
        if verbose:
            print(f"Warning: Error in final cleanup: {e}")
    
    if verbose:
        final_shape = df_clean.shape
        final_memory = df_clean.memory_usage(deep=True).sum()
        memory_reduction = (1 - final_memory / original_memory) * 100
        
        print(f"\nCleanup complete!")
        print(f"Shape: {original_shape} -> {final_shape}")
        print(f"Memory usage: {original_memory / 1024**2:.2f} MB -> {final_memory / 1024**2:.2f} MB")
        print(f"Memory reduction: {memory_reduction:.1f}%")
        print(f"Data types summary:")
        print(df_clean.dtypes.value_counts().to_string())
    
    return df_clean


# Enhanced convenience functions with better defaults
def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quick clean with minimal processing for fast results"""
    return deep_clean_dataframe(
        df, 
        missing_threshold=0.8,
        numeric_conversion_threshold=0.95,
        standardize_text=True,
        handle_outliers=False,
        remove_duplicates=True,
        categorical_threshold=5,
        verbose=False
    )

def thorough_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Thorough clean with all features enabled for comprehensive cleaning"""
    return deep_clean_dataframe(
        df,
        missing_threshold=0.3,
        numeric_conversion_threshold=0.8,
        categorical_threshold=15,
        standardize_text=True,
        handle_outliers=True,
        outlier_method='iqr',
        remove_duplicates=True,
        verbose=True
    )

def conservative_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative clean that preserves more data with minimal changes"""
    return deep_clean_dataframe(
        df,
        missing_threshold=0.9,
        numeric_conversion_threshold=0.95,
        categorical_threshold=20,
        standardize_text=False,
        handle_outliers=False,
        remove_duplicates=True,
        verbose=False
    )