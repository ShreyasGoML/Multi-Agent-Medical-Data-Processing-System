"""
Medical Knowledge Agent for Multi-Agent Medical Data Processing System

This agent validates and standardizes medical terminology in datasets
by integrating with external medical knowledge APIs.
"""

import pandas as pd
from typing import Dict, Any, List, Optional, Set
import re
import requests
import os


def call_perplexity_api(term: str) -> Optional[str]:
    """
    Call Perplexity API to validate and standardize medical terminology.
    
    Args:
        term (str): The medical term to validate and standardize
        
    Returns:
        Optional[str]: Standardized version of the term if found, None otherwise
    """
    api_key = os.getenv('PERPLEXITY_API_KEY')
    if not api_key:
        print("Warning: PERPLEXITY_API_KEY not found in environment variables")
        return None
    
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        prompt = f"""
        Validate and standardize the medical term: "{term}"
        
        Provide only the standardized medical term if it exists.
        If the term is already correct, return the same term.
        If the term is not a valid medical term, return "INVALID".
        
        Response format: [standardized_term]
        """
        
        data = {
            "model": "llama-3.1-sonar-small-128k-online",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = requests.post(
            'https://api.perplexity.ai/chat/completions',
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            standardized_term = result['choices'][0]['message']['content'].strip()
            
            if standardized_term and standardized_term != "INVALID":
                return standardized_term
                
    except Exception as e:
        print(f"Error calling Perplexity API for term '{term}': {str(e)}")
    
    return None


def extract_medical_terms(dataframe: pd.DataFrame, target_columns: List[str]) -> Set[str]:
    """
    Extract unique medical terms from specified columns in the dataframe.
    
    Args:
        dataframe (pd.DataFrame): The medical dataset
        target_columns (List[str]): List of column names to extract terms from
        
    Returns:
        Set[str]: Set of unique medical terms found across all target columns
    """
    all_terms = set()
    
    for column in target_columns:
        if column in dataframe.columns:
            
            column_values = dataframe[column].dropna().astype(str)
            
            for value in column_values:
                # Split on common delimiters and clean terms
                terms = re.split(r'[,;|\n\r]+', value)
                for term in terms:
                    # Clean the term: remove extra whitespace, special characters
                    cleaned_term = re.sub(r'[^\w\s-]', '', term.strip())
                    if cleaned_term and len(cleaned_term) > 1:  # Ignore single characters
                        all_terms.add(cleaned_term)
    
    return all_terms


def categorize_term_issues(original_term: str, standardized_term: str) -> str:
    """
    Categorize the type of issue found with a medical term.
    
    Args:
        original_term (str): The original term from the dataset
        standardized_term (str): The standardized term from the API
        
    Returns:
        str: Category of the issue ('corrections', 'standardizations', or 'expansions')
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


def validate_medical_terminology(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate, standardize, and expand medical terminology from a medical dataset.
    This function acts as a node in a LangGraph workflow.
    
    Args:
        state (Dict[str, Any]): State dictionary containing the dataset under 'dataframe' key
        
    Returns:
        Dict[str, Any]: Updated state dictionary with 'terminology_report' added
        
    Raises:
        KeyError: If 'dataframe' key is not found in the state
        ValueError: If the dataframe is empty or invalid
    """
    
    # Validate input state
    if 'dataframe' not in state:
        raise KeyError("State dictionary must contain 'dataframe' key")
    
    dataframe = state['dataframe']
    
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("'dataframe' value must be a pandas DataFrame")
    
    if dataframe.empty:
        raise ValueError("DataFrame cannot be empty")
    
    # Define target columns that typically contain medical terminology
    target_columns = ['test', 'biomarker', 'chief_remark', 'provisionaldiagnosis', 'finaldiagnosis']
    
    print(f"Analyzing medical terminology in columns: {target_columns}")
    
    unique_terms = extract_medical_terms(dataframe, target_columns)
    print(f"Found {len(unique_terms)} unique medical terms to validate")
    
    terminology_report = {
        'corrections': {},      # Spelling corrections
        'standardizations': {}, # Terminology standardizations
        'expansions': {}        # Abbreviation expansions
    }
    
    processed_count = 0
    for term in unique_terms:
        standardized_term = call_perplexity_api(term)
        
        if standardized_term and standardized_term.lower() != term.lower():
            category = categorize_term_issues(term, standardized_term)
            terminology_report[category][term] = standardized_term
            processed_count += 1
        
        if processed_count % 10 == 0 and processed_count > 0:
            print(f"Processed {processed_count} terms with issues found")

    # Summary statistics
    total_issues = sum(len(category) for category in terminology_report.values())
    print(f"Terminology validation complete:")
    print(f"  - Spelling corrections: {len(terminology_report['corrections'])}")
    print(f"  - Standardizations: {len(terminology_report['standardizations'])}")
    print(f"  - Abbreviation expansions: {len(terminology_report['expansions'])}")
    print(f"  - Total issues found: {total_issues}")
    
    # Update state with terminology report
    state['terminology_report'] = terminology_report
    
    return state


