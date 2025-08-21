import json
from openai import OpenAI
from config import OPENAI_API_KEY, CODE_GENERATION_PROMPT, OPENAI_MODEL

class CodeGenerationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Code Generation Agent"
    
    def generate_code(self, cleaning_strategy: dict) -> str:
        prompt = f'''
{CODE_GENERATION_PROMPT}

Cleaning Strategy:
{json.dumps(cleaning_strategy, indent=2)}

Generate executable Python code that:
1. Implements all recommended cleaning strategies
2. Uses pandas for data manipulation
3. Includes error handling and logging
4. Preserves data integrity
5. Is production-ready

Return ONLY the Python code, no explanations.
'''
        
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"# Error generating code: {str(e)}\nprint('Code generation failed')"
