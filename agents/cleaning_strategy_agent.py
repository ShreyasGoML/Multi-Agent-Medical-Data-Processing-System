import json
from openai import OpenAI
from config import OPENAI_API_KEY, CLEANING_STRATEGY_PROMPT, OPENAI_MODEL

class CleaningStrategyAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Cleaning Strategy Agent"
    
    def plan(self, analysis_result: dict, validation_result: dict) -> dict:
        # Extract medical terminology fixes from validation result
        medical_fixes = {}
        if 'validation_result' in validation_result and 'terminology_report' in validation_result['validation_result']:
            terminology_report = validation_result['validation_result']['terminology_report']
            
            # Combine all medical terminology fixes
            medical_fixes = {
                **terminology_report.get('corrections', {}),
                **terminology_report.get('standardizations', {}),
                **terminology_report.get('expansions', {})
            }
        
        prompt = f'''
        {CLEANING_STRATEGY_PROMPT}
        
        Data Analysis Results:
        {json.dumps(analysis_result, indent=2)}
        
        Medical Validation Results:
        {json.dumps(validation_result, indent=2)}
        
        Medical Terminology Fixes Available:
        {json.dumps(medical_fixes, indent=2)}
        
        Provide comprehensive cleaning strategy in JSON format with:
        - missing_value_strategies: Column-specific imputation methods
        - outlier_handling: Methods for each numeric column
        - text_standardization: Approaches for text columns
        - duplicate_removal: Strategy for handling duplicates
        - data_type_corrections: Type conversion recommendations
        - medical_terminology_fixes: Dictionary mapping original terms to corrected terms
        - priority_order: Order of cleaning operations
        '''
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            strategy = json.loads(response.choices[0].message.content)
            
            # Ensure medical terminology fixes are included
            if medical_fixes:
                strategy['medical_terminology_fixes'] = medical_fixes
            
            return {
                "cleaning_strategy": strategy,
                "agent": self.name
            }
        except Exception as e:
            # Fallback strategy with medical fixes
            fallback_strategy = {
                "missing_value_strategies": {},
                "outlier_handling": {},
                "text_standardization": {},
                "duplicate_removal": False,
                "data_type_corrections": {},
                "medical_terminology_fixes": medical_fixes,
                "priority_order": ["medical_terminology_fixes", "missing_values", "duplicates", "outliers"]
            }
            
            return {
                "cleaning_strategy": fallback_strategy,
                "agent": self.name,
                "error": str(e)
            }
