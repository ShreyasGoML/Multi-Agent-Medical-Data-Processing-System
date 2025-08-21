import json
from openai import OpenAI
from config import OPENAI_API_KEY, CLEANING_STRATEGY_PROMPT, OPENAI_MODEL

class CleaningStrategyAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Cleaning Strategy Agent"
    
    def plan(self, analysis_result: dict, validation_result: dict) -> dict:
        prompt = f'''
{CLEANING_STRATEGY_PROMPT}

Data Analysis Results:
{json.dumps(analysis_result, indent=2)}

Medical Validation Results:
{json.dumps(validation_result, indent=2)}

Provide comprehensive cleaning strategy in JSON format with:
- missing_value_strategies: Column-specific imputation methods
- outlier_handling: Methods for each numeric column
- text_standardization: Approaches for text columns
- duplicate_removal: Strategy for handling duplicates
- data_type_corrections: Type conversion recommendations
- priority_order: Order of cleaning operations
'''
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            strategy = json.loads(response.choices[0].message.content)
            
            return {
                "cleaning_strategy": strategy,
                "agent": self.name
            }
        except Exception as e:
            return {
                "cleaning_strategy": {"error": str(e)},
                "agent": self.name
            }
