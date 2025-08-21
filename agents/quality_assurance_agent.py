import pandas as pd
import json
from openai import OpenAI
from config import OPENAI_API_KEY, QA_PROMPT, OPENAI_MODEL

class QualityAssuranceAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Quality Assurance Agent"
    
    def validate(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
        comparison_stats = {
            "original_shape": original_df.shape,
            "cleaned_shape": cleaned_df.shape,
            "rows_removed": original_df.shape[0] - cleaned_df.shape[0],
            "columns_modified": list(set(original_df.columns) - set(cleaned_df.columns)),
            "missing_reduction": {
                col: {
                    "original": int(original_df[col].isnull().sum()),
                    "cleaned": int(cleaned_df[col].isnull().sum()) if col in cleaned_df.columns else 0
                }
                for col in original_df.columns
            }
        }
        
        prompt = f'''
{QA_PROMPT}

Comparison Statistics:
{json.dumps(comparison_stats, indent=2)}

Provide comprehensive QA assessment in JSON format with:
- data_integrity_score: 0-100 score
- cleaning_effectiveness: Assessment of each cleaning operation
- potential_issues: Any concerns with the cleaning process
- recommendations: Suggestions for improvement
- approval_status: APPROVED/NEEDS_REVIEW/REJECTED
'''
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            qa_result = json.loads(response.choices[0].message.content)
            
            return {
                "comparison_stats": comparison_stats,
                "qa_result": qa_result,
                "agent": self.name
            }
        except Exception as e:
            return {
                "comparison_stats": comparison_stats,
                "qa_result": {"error": str(e)},
                "agent": self.name
            }
