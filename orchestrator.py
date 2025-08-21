import pandas as pd
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from agents.data_analysis_agent import DataAnalysisAgent
from agents.medical_knowledge_agent import MedicalKnowledgeAgent
from agents.cleaning_strategy_agent import CleaningStrategyAgent
from agents.code_generation_agent import CodeGenerationAgent
from agents.quality_assurance_agent import QualityAssuranceAgent

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    dataframe: pd.DataFrame
    analysis_result: Dict[str, Any]
    validation_result: Dict[str, Any]
    cleaning_strategy: Dict[str, Any]
    generated_code: str
    cleaned_dataframe: pd.DataFrame
    qa_result: Dict[str, Any]
    current_step: str

class MedicalDataOrchestrator:
    def __init__(self):
        self.data_analysis_agent = DataAnalysisAgent()
        self.medical_knowledge_agent = MedicalKnowledgeAgent()
        self.cleaning_strategy_agent = CleaningStrategyAgent()
        self.code_generation_agent = CodeGenerationAgent()
        self.qa_agent = QualityAssuranceAgent()
        
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("data_analysis", self._run_data_analysis)
        workflow.add_node("medical_validation", self._run_medical_validation)
        workflow.add_node("cleaning_strategy", self._run_cleaning_strategy)
        workflow.add_node("code_generation", self._run_code_generation)
        workflow.add_node("execute_cleaning", self._execute_cleaning)
        workflow.add_node("quality_assurance", self._run_quality_assurance)
        
        # Define the workflow edges
        workflow.set_entry_point("data_analysis")
        workflow.add_edge("data_analysis", "medical_validation")
        workflow.add_edge("medical_validation", "cleaning_strategy")
        workflow.add_edge("cleaning_strategy", "code_generation")
        workflow.add_edge("code_generation", "execute_cleaning")
        workflow.add_edge("execute_cleaning", "quality_assurance")
        workflow.add_edge("quality_assurance", END)
        
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
        return state
    
    def _run_cleaning_strategy(self, state: AgentState) -> AgentState:
        print("🧹 Developing Cleaning Strategy...")
        cleaning_strategy = self.cleaning_strategy_agent.plan(
            state["analysis_result"], 
            state["validation_result"]
        )
        state["cleaning_strategy"] = cleaning_strategy
        state["current_step"] = "Cleaning Strategy Complete"
        state["messages"].append(f"Cleaning strategy developed by {cleaning_strategy['agent']}")
        return state
    
    def _run_code_generation(self, state: AgentState) -> AgentState:
        print("💻 Generating Cleaning Code...")
        generated_code = self.code_generation_agent.generate_code(
            state["cleaning_strategy"]
        )
        state["generated_code"] = generated_code
        state["current_step"] = "Code Generation Complete"
        state["messages"].append("Cleaning code generated successfully")
        return state
    
    def _execute_cleaning(self, state: AgentState) -> AgentState:
        print("⚙️ Executing Cleaning Operations...")
        try:
            # Execute the generated code
            local_vars = {"df": state["dataframe"].copy(), "pd": pd}
            exec(state["generated_code"], {"__builtins__": {}}, local_vars)
            state["cleaned_dataframe"] = local_vars["df"]
            state["current_step"] = "Data Cleaning Complete"
            state["messages"].append("Data cleaning executed successfully")
        except Exception as e:
            print(f"Error executing cleaning code: {str(e)}")
            state["cleaned_dataframe"] = state["dataframe"].copy()
            state["messages"].append(f"Cleaning execution failed: {str(e)}")
        return state
    
    def _run_quality_assurance(self, state: AgentState) -> AgentState:
        print("✅ Running Quality Assurance...")
        qa_result = self.qa_agent.validate(
            state["dataframe"], 
            state["cleaned_dataframe"]
        )
        state["qa_result"] = qa_result
        state["current_step"] = "Quality Assurance Complete"
        state["messages"].append(f"Quality assurance completed by {qa_result['agent']}")
        return state
    
    def process_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Main entry point for processing medical data"""
        initial_state = AgentState(
            messages=[],
            dataframe=df,
            analysis_result={},
            validation_result={},
            cleaning_strategy={},
            generated_code="",
            cleaned_dataframe=pd.DataFrame(),
            qa_result={},
            current_step="Starting"
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        return {
            "original_dataframe": final_state["dataframe"],
            "analysis_result": final_state["analysis_result"],
            "validation_result": final_state["validation_result"],
            "cleaning_strategy": final_state["cleaning_strategy"],
            "generated_code": final_state["generated_code"],
            "cleaned_dataframe": final_state["cleaned_dataframe"],
            "qa_result": final_state["qa_result"],
            "messages": final_state["messages"]
        }
