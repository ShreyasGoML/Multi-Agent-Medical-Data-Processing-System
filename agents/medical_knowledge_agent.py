import pandas as pd
import json
from openai import OpenAI
from config import OPENAI_API_KEY, MEDICAL_KNOWLEDGE_PROMPT, OPENAI_MODEL

class MedicalKnowledgeAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Medical Knowledge Agent"
    
    def validate(self, df: pd.DataFrame) -> dict:
        medical_columns = ['biomarker', 'provisionaldiagnosis', 'finaldiagnosis', 
                          'chief_remark', 'clinical_note', 'vital_remark']
        
        medical_data = {}
        for col in medical_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()[:20]  # Limit for API
                medical_data[col] = list(unique_values)
        
        prompt = f'''
{MEDICAL_KNOWLEDGE_PROMPT}

Medical Data to Validate:
{json.dumps(medical_data, indent=2)}

Provide validation results in JSON format with:
- unknown_terms: List of unrecognized medical terms
- misspelled_terms: Likely misspellings with suggestions
- inconsistent_coding: Inconsistent medical codes/formats
- suspicious_values: Clinically suspicious values
'''
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            validation_result = json.loads(response.choices[0].message.content)
            
            return {
                "validation_result": validation_result,
                "agent": self.name
            }
        except Exception as e:
            return {
                "validation_result": {"error": str(e)},
                "agent": self.name
            }
