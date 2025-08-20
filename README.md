# Multi-Agent-Medical-Data-Processing-System

Starter template to build the application
```
medical_multi_agent_system/
├── requirements.txt
├── config.py
├── .env.example
├── orchestrator.py
├── streamlit_app.py
├── README.md
└── agents/
    ├── __init__.py
    ├── data_analysis_agent.py
    ├── medical_knowledge_agent.py
    ├── cleaning_strategy_agent.py
    ├── code_generation_agent.py
    └── quality_assurance_agent.py
```

## Agent Architecture
The system uses **Langraph** to orchestrate 5 specialized agents:

1. **Data Analysis Agent** - Analyzes data quality and identifies issues
2. **Medical Knowledge Agent** - Validates medical terminology and clinical data  
3. **Cleaning Strategy Agent** - Develops optimal cleaning strategies
4. **Code Generation Agent** - Creates executable Python cleaning code
5. **Quality Assurance Agent** - Validates results and ensures data integrity

## Usage
1. Upload your medical dataset (CSV/Excel)
2. Click "Start Multi-Agent Processing"
3. Review agent results in organized tabs
4. Download cleaned data and generated code
5. Interact with agents via the query interface

## Requirements
- Python 3.8+
- OpenAI API key
