import pandas as pd
import json
from typing import Dict, Any, List, Optional, Set
import re
import os
from openai import OpenAI
from config import OPENAI_API_KEY, MEDICAL_KNOWLEDGE_PROMPT, OPENAI_MODEL

class MedicalKnowledgeAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.name = "Medical Knowledge Agent"
    
    def call_openai_agent(self, term: str) -> Optional[str]:
        """
        Call OpenAI Agent to validate and standardize medical terminology.
        
        Args:
            term (str): The medical term to validate and standardize
        
        Returns:
            Optional[str]: Standardized version of the term if found, None otherwise
        """
        try:
            # Prompt given to the agent
            prompt = f"""
            Validate and standardize the medical term: "{term}"
            
            Provide only the standardized medical term if it exists.
            If the term is already correct, return the same term.
            If the term is not a valid medical term, return "INVALID".
            
            Response format: [standardized_term]
            """
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            standardized_term = response.choices[0].message.content.strip()
            
            if standardized_term and standardized_term != "INVALID":
                return standardized_term
                
        except Exception as e:
            print(f"Error calling OpenAI Agent for term '{term}': {str(e)}")
        
        return None

    def extract_medical_terms(self, dataframe: pd.DataFrame, target_columns: List[str]) -> Set[str]:
        """
        Extract unique medical terms from specified columns in the dataframe.
        """
        all_terms = set()
        
        for column in target_columns:
            if column in dataframe.columns:
                column_values = dataframe[column].dropna().astype(str)
                for value in column_values:
                    terms = re.split(r'[,;|\n\r]+', value)
                    for term in terms:
                        cleaned_term = re.sub(r'[^\w\s-]', '', term.strip())
                        if cleaned_term and len(cleaned_term) > 1:
                            all_terms.add(cleaned_term)
        
        return all_terms

    def categorize_term_issues(self, original_term: str, standardized_term: str) -> str:
        """
        Categorize the type of issue found with a medical term.
        """
        original_lower = original_term.lower()
        standardized_lower = standardized_term.lower()
        
        if len(original_lower) > 2 and len(standardized_lower) > 2:
            common_chars = sum(1 for a, b in zip(original_lower, standardized_lower) if a == b)
            if abs(len(original_lower) - len(standardized_lower)) <= 2 and common_chars >= len(original_lower) * 0.7:
                return 'corrections'
        
        if len(original_lower) <= 4 and len(standardized_lower) > len(original_lower) * 2:
            return 'expansions'
        
        return 'standardizations'

    def validate_medical_terminology(self, dataframe: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate, standardize, and expand medical terminology from a medical dataset.
        """
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty")
        
        # Target columns
        target_columns = ['test', 'biomarker', 'chief_remark', 'provisionaldiagnosis', 'finaldiagnosis']
        print(f"Analyzing medical terminology in columns: {target_columns}")
        
        unique_terms = self.extract_medical_terms(dataframe, target_columns)
        print(f"Found {len(unique_terms)} unique medical terms to validate")
        
        terminology_report = {
            'corrections': {},
            'standardizations': {},
            'expansions': {}
        }
        
        processed_count = 0
        for term in unique_terms:
            standardized_term = self.call_openai_agent(term)
            
            if standardized_term and standardized_term.lower() != term.lower():
                category = self.categorize_term_issues(term, standardized_term)
                terminology_report[category][term] = standardized_term
                processed_count += 1
            
            if processed_count % 10 == 0 and processed_count > 0:
                print(f"Processed {processed_count} terms with issues found")
        
        # Stats
        total_issues = sum(len(category) for category in terminology_report.values())
        print("Terminology validation complete:")
        print(f"  - Spelling corrections: {len(terminology_report['corrections'])}")
        print(f"  - Standardizations: {len(terminology_report['standardizations'])}")
        print(f"  - Abbreviation expansions: {len(terminology_report['expansions'])}")
        print(f"  - Total issues found: {total_issues}")
        
        return terminology_report
    
    def validate(self, df: pd.DataFrame) -> dict:
        """
        Main validate method to maintain compatibility with existing interface
        """
        try:
            # Use the new terminology validation logic
            terminology_report = self.validate_medical_terminology(df)
            
            # Format for compatibility with existing system
            validation_result = {
                "terminology_report": terminology_report,
                "summary": {
                    "corrections_count": len(terminology_report.get('corrections', {})),
                    "standardizations_count": len(terminology_report.get('standardizations', {})),
                    "expansions_count": len(terminology_report.get('expansions', {})),
                    "total_issues": sum(len(category) for category in terminology_report.values())
                }
            }
            
            return {
                "validation_result": validation_result,
                "agent": self.name
            }
            
        except Exception as e:
            return {
                "validation_result": {"error": str(e)},
                "agent": self.name
            }
