import pandas as pd
import numpy as np
import re
from typing import Dict, Any
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL

class DataCleaningAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Data Cleaning Agent"
    
    def apply_medical_terminology_fixes(self, df: pd.DataFrame, medical_fixes: dict) -> pd.DataFrame:
        """Apply medical terminology corrections to the dataframe"""
        if not medical_fixes:
            return df
        
        print(f"Applying {len(medical_fixes)} medical terminology corrections...")
        
        # Target columns for medical terminology
        medical_columns = ['test', 'biomarker', 'chief_remark', 'provisionaldiagnosis', 'finaldiagnosis']
        corrections_applied = 0
        
        for column in medical_columns:
            if column in df.columns:
                original_values = df[column].copy()
                
                for original_term, corrected_term in medical_fixes.items():
                    # Apply regex replacement with word boundaries and case insensitive
                    pattern = r'\b' + re.escape(str(original_term)) + r'\b'
                    df[column] = df[column].astype(str).str.replace(
                        pattern, str(corrected_term), regex=True, case=False
                    )
                
                # Count actual changes made
                changes_in_column = (original_values != df[column]).sum()
                if changes_in_column > 0:
                    corrections_applied += changes_in_column
                    print(f"Applied {changes_in_column} corrections in {column} column")
        
        print(f"Total medical terminology corrections applied: {corrections_applied}")
        return df
    
    def clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values appropriately"""
        print("Cleaning missing values...")
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                if df[column].dtype in ['object', 'string']:
                    # For text columns, fill with 'Unknown' or most frequent value
                    mode_value = df[column].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                    df[column] = df[column].fillna(fill_value)
                else:
                    # For numeric columns, fill with median
                    median_value = df[column].median()
                    df[column] = df[column].fillna(median_value)
                
                print(f"Filled {missing_count} missing values in {column}")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        original_count = len(df)
        df = df.drop_duplicates()
        removed_count = original_count - len(df)
        
        if removed_count > 0:
            print(f"Removed {removed_count} duplicate rows")
        else:
            print("No duplicates found")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numeric columns using IQR method"""
        print("Handling outliers in numeric columns...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:  # Avoid division by zero
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = len(df[(df[column] < lower_bound) | (df[column] > upper_bound)])
                
                if outliers_count > 0:
                    # Cap outliers instead of removing them
                    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                    print(f"Capped {outliers_count} outliers in {column}")
        
        return df
    
    def standardize_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text formatting"""
        print("Standardizing text columns...")
        
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        
        for column in text_columns:
            # Remove extra whitespaces and standardize case where appropriate
            df[column] = df[column].astype(str).str.strip()
            
            # Remove multiple consecutive spaces
            df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
            
            # Clean special characters for specific columns
            if column in ['patient_id', 'test', 'biomarker']:
                df[column] = df[column].str.replace(r'[^\w\s\-;,()]', '', regex=True)
        
        return df
    
    def clean_data(self, original_df: pd.DataFrame, analysis_result: dict, medical_validation: dict) -> dict:
        """Main cleaning function that applies all cleaning operations"""
        
        # Create a copy to work with
        df = original_df.copy()
        cleaning_summary = []
        
        print("🧹 Starting comprehensive data cleaning...")
        
        # Store original metrics - FIXED: Access tuple elements properly
        original_shape = original_df.shape
        original_rows = int(original_shape[0])  # Extract int from tuple
        original_cols = int(original_shape[1])  # Extract int from tuple
        original_missing = int(original_df.isnull().sum().sum())
        
        # 1. Apply medical terminology fixes first
        medical_fixes = {}
        if 'validation_result' in medical_validation and 'terminology_report' in medical_validation['validation_result']:
            terminology_report = medical_validation['validation_result']['terminology_report']
            medical_fixes = {
                **terminology_report.get('corrections', {}),
                **terminology_report.get('standardizations', {}),
                **terminology_report.get('expansions', {})
            }
        
        if medical_fixes:
            df = self.apply_medical_terminology_fixes(df, medical_fixes)
            cleaning_summary.append(f"Applied {len(medical_fixes)} medical terminology corrections")
        
        # 2. Standardize text formatting
        df = self.standardize_text_columns(df)
        cleaning_summary.append("Standardized text formatting")
        
        # 3. Handle missing values
        df = self.clean_missing_values(df)
        final_missing = int(df.isnull().sum().sum())
        cleaning_summary.append(f"Reduced missing values from {original_missing} to {final_missing}")
        
        # 4. Remove duplicates
        df = self.remove_duplicates(df)
        final_shape = df.shape
        final_rows = int(final_shape[0])  # Extract int from tuple
        final_cols = int(final_shape[1])  # Extract int from tuple
        cleaning_summary.append(f"Removed {original_rows - final_rows} duplicate rows")
        
        # 5. Handle outliers
        df = self.handle_outliers(df)
        cleaning_summary.append("Applied outlier capping to numeric columns")
        
        print("✅ Data cleaning completed successfully!")
        
        # Generate quality report - FIXED: Use extracted integers
        quality_report = {
            "original_shape": (original_rows, original_cols),
            "cleaned_shape": (final_rows, final_cols),
            "missing_values_reduced": original_missing - final_missing,
            "duplicates_removed": original_rows - final_rows,
            "medical_corrections_applied": len(medical_fixes),
            "cleaning_summary": cleaning_summary,
            "data_quality_score": self._calculate_quality_score(df, original_df, original_rows, final_rows)
        }
        
        return {
            "cleaned_dataframe": df,
            "quality_report": quality_report,
            "medical_fixes_applied": medical_fixes,
            "agent": self.name
        }
    
    def _calculate_quality_score(self, cleaned_df: pd.DataFrame, original_df: pd.DataFrame, 
                                original_rows: int, final_rows: int) -> int:
        """Calculate a quality score from 0-100 - FIXED: Use integers for calculations"""
        score = 100
        
        # Deduct points for remaining missing values
        total_cells = cleaned_df.size
        if total_cells > 0:
            missing_percentage = (cleaned_df.isnull().sum().sum() / total_cells) * 100
            score -= missing_percentage * 2
        
        # Deduct points for data loss (if significant)
        if original_rows > 0:
            data_loss_percentage = ((original_rows - final_rows) / original_rows) * 100
            if data_loss_percentage > 10:  # Only penalize if more than 10% data loss
                score -= data_loss_percentage
        
        return max(0, min(100, int(score)))
