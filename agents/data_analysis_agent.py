import pandas as pd
import json
from openai import OpenAI
from config import OPENAI_API_KEY, DATA_ANALYSIS_PROMPT, OPENAI_MODEL

class DataAnalysisAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Data Analysis Agent"
    
    def analyze(self, df: pd.DataFrame) -> dict:
        basic_stats = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_counts": df.isnull().sum().to_dict(),
            "duplicate_count": int(df.duplicated().sum()),
            "memory_usage": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        sample_data = df.head(10).to_dict('records')
        
        prompt = f"""
{DATA_ANALYSIS_PROMPT}

Dataset Info:
- Shape: {basic_stats['shape']}
- Columns: {basic_stats['columns']}
- Missing Values: {basic_stats['missing_counts']}
- Duplicates: {basic_stats['duplicate_count']}

Sample Data:
{json.dumps(sample_data, indent=2)}

Provide detailed analysis in JSON format.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            llm_analysis = json.loads(response.choices[0].message.content)
            
            return {
                "basic_stats": basic_stats,
                "llm_analysis": llm_analysis,
                "agent": self.name
            }
        except Exception as e:
            return {
                "basic_stats": basic_stats,
                "llm_analysis": {"error": str(e)},
                "agent": self.name
            }
