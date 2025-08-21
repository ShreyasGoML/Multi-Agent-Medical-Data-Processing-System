import json
from openai import OpenAI
from config import OPENAI_API_KEY, CODE_GENERATION_PROMPT, OPENAI_MODEL

class CodeGenerationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Code Generation Agent"
    
    def generate_medical_terminology_code(self, medical_fixes: dict) -> str:
        """Generate code to apply medical terminology fixes"""
        if not medical_fixes:
            return "# No medical terminology fixes needed"
        
        code_lines = [
            "# Apply Medical Terminology Fixes",
            "print('Applying medical terminology corrections...')",
            ""
        ]
        
        # Generate replacement code for medical terminology columns
        medical_columns = ['test', 'biomarker', 'chief_remark', 'provisionaldiagnosis', 'finaldiagnosis']
        
        for column in medical_columns:
            code_lines.append(f"# Fix terminology in {column} column")
            code_lines.append(f"if '{column}' in df.columns:")
            
            for original_term, corrected_term in medical_fixes.items():
                # Use regex for partial matches and word boundaries
                code_lines.append(f"    df['{column}'] = df['{column}'].astype(str).str.replace(r'\\b{original_term}\\b', '{corrected_term}', regex=True, case=False)")
            
            code_lines.append("")
        
        code_lines.append("print(f'Applied {len(medical_fixes)} medical terminology corrections')")
        code_lines.append("")
        
        return "\n".join(code_lines)
    
    def generate_code(self, cleaning_strategy: dict) -> str:
        try:
            # Extract medical terminology fixes
            medical_fixes = cleaning_strategy.get('cleaning_strategy', {}).get('medical_terminology_fixes', {})
            
            # Generate medical terminology correction code
            medical_code = self.generate_medical_terminology_code(medical_fixes)
            
            prompt = f'''
            {CODE_GENERATION_PROMPT}
            
            Cleaning Strategy:
            {json.dumps(cleaning_strategy, indent=2)}
            
            Generate executable Python code that:
            1. Applies medical terminology corrections first
            2. Implements all other recommended cleaning strategies
            3. Uses pandas for data manipulation
            4. Includes error handling and logging
            5. Preserves data integrity
            6. Is production-ready
            
            Start with this medical terminology correction code:
            {medical_code}
            
            Then add other cleaning operations. Return ONLY the complete Python code.
            '''
            
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            generated_code = response.choices[0].message.content
            
            # Ensure medical terminology fixes are included at the beginning
            if medical_fixes and medical_code not in generated_code:
                generated_code = medical_code + "\n\n" + generated_code
            
            return generated_code
            
        except Exception as e:
            # Fallback: Generate basic cleaning code with medical fixes
            medical_fixes = cleaning_strategy.get('cleaning_strategy', {}).get('medical_terminology_fixes', {})
            medical_code = self.generate_medical_terminology_code(medical_fixes)
            
            fallback_code = f"""
import pandas as pd
import numpy as np
import re

print("Starting data cleaning process...")

{medical_code}

# Handle missing values
print("Handling missing values...")
for col in df.columns:
    if df[col].dtype in ['object', 'string']:
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].median())

# Remove duplicates
print("Removing duplicates...")
original_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {{original_rows - len(df)}} duplicate rows")

# Handle outliers in numeric columns
print("Handling outliers...")
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

print("Data cleaning completed successfully!")
print(f"Final dataset shape: {{df.shape}}")
"""
            return fallback_code
