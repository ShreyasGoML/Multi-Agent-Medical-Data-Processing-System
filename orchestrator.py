import pandas as pd
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from agents.data_analysis_agent import DataAnalysisAgent
from agents.medical_knowledge_agent import MedicalKnowledgeAgent
from agents.data_cleaning_agent import DataCleaningAgent

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    dataframe: pd.DataFrame
    analysis_result: Dict[str, Any]
    validation_result: Dict[str, Any]
    cleaning_result: Dict[str, Any]
    current_step: str

class MedicalDataOrchestrator:
    def __init__(self):
        self.data_analysis_agent = DataAnalysisAgent()
        self.medical_knowledge_agent = MedicalKnowledgeAgent()
        self.data_cleaning_agent = DataCleaningAgent()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent (simplified to 3 agents)
        workflow.add_node("data_analysis", self._run_data_analysis)
        workflow.add_node("medical_validation", self._run_medical_validation)
        workflow.add_node("data_cleaning", self._run_data_cleaning)
        
        # Define the simplified workflow edges
        workflow.set_entry_point("data_analysis")
        workflow.add_edge("data_analysis", "medical_validation")
        workflow.add_edge("medical_validation", "data_cleaning")
        workflow.add_edge("data_cleaning", END)
        
        return workflow.compile()
    
    def _run_data_analysis(self, state: AgentState) -> AgentState:
        print("🔍 Running Data Analysis...")
        analysis_result = self.data_analysis_agent.analyze(state["dataframe"])
        state["analysis_result"] = analysis_result
        state["current_step"] = "Data Analysis Complete"
        state["messages"].append(f"Data analysis completed by {analysis_result['agent']}")
        return state
    
    def _run_medical_validation(self, state: AgentState) -> AgentState:
        print("🏥 Running Medical Knowledge Validation...")
        validation_result = self.medical_knowledge_agent.validate(state["dataframe"])
        state["validation_result"] = validation_result
        state["current_step"] = "Medical Validation Complete"
        state["messages"].append(f"Medical validation completed by {validation_result['agent']}")
        
        # Log medical terminology findings
        if 'validation_result' in validation_result and 'terminology_report' in validation_result['validation_result']:
            terminology_report = validation_result['validation_result']['terminology_report']
            total_fixes = sum(len(category) for category in terminology_report.values())
            state["messages"].append(f"Identified {total_fixes} medical terminology issues for correction")
        
        return state
    
    def _run_data_cleaning(self, state: AgentState) -> AgentState:
        print("🧹 Running Data Cleaning...")
        cleaning_result = self.data_cleaning_agent.clean_data(
            state["dataframe"],
            state["analysis_result"],
            state["validation_result"]
        )
        state["cleaning_result"] = cleaning_result
        state["current_step"] = "Data Cleaning Complete"
        state["messages"].append(f"Data cleaning completed by {cleaning_result['agent']}")
        
        # Log cleaning summary
        quality_report = cleaning_result.get("quality_report", {})
        score = quality_report.get("data_quality_score", 0)
        state["messages"].append(f"Data quality score: {score}/100")
        
        return state
    
    def process_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main entry point for processing medical data"""
        initial_state = AgentState(
            messages=[],
            dataframe=df,
            analysis_result={},
            validation_result={},
            cleaning_result={},
            current_step="Starting"
        )
        
        # Run the simplified workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "original_dataframe": final_state["dataframe"],
            "analysis_result": final_state["analysis_result"],
            "validation_result": final_state["validation_result"],
            "cleaning_result": final_state["cleaning_result"],
            "cleaned_dataframe": final_state["cleaning_result"]["cleaned_dataframe"],
            "messages": final_state["messages"]
        }
