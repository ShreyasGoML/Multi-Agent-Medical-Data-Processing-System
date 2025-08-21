import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from orchestrator import MedicalDataOrchestrator

st.set_page_config(
    page_title="GoML Multi-Agent Medical Data Cleaning System",
    page_icon="🏥",
    layout="wide"
)

def process_user_query(query: str, df: pd.DataFrame = None, results: dict = None):
    """Process user query and route to appropriate agents"""
    
    # Initialize orchestrator and agents
    orchestrator = MedicalDataOrchestrator()
    
    # Determine which agent should handle the query based on keywords
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['analyze', 'analysis', 'missing', 'duplicate', 'quality']):
        if df is not None:
            agent_response = orchestrator.data_analysis_agent.analyze(df)
            return f"**Data Analysis Agent Response:**\n{json.dumps(agent_response, indent=2)}"
    
    elif any(word in query_lower for word in ['medical', 'terminology', 'diagnosis', 'biomarker', 'clinical']):
        if df is not None:
            agent_response = orchestrator.medical_knowledge_agent.validate(df)
            return f"**Medical Knowledge Agent Response:**\n{json.dumps(agent_response, indent=2)}"
    
    elif any(word in query_lower for word in ['clean', 'cleaning', 'fix', 'correct']):
        if results and 'analysis_result' in results and 'validation_result' in results:
            agent_response = orchestrator.data_cleaning_agent.clean_data(
                df, results['analysis_result'], results['validation_result']
            )
            return f"**Data Cleaning Agent Response:**\n{json.dumps(agent_response['quality_report'], indent=2)}"
    
    # General query - route to most appropriate agent or provide guidance
    return f"""
    **Multi-Agent System Response:**
    
    Your query: "{query}"
    
    I can help you with:
    - **Data Analysis**: Ask about missing values, duplicates, data quality
    - **Medical Knowledge**: Ask about medical terminology, diagnoses, biomarkers  
    - **Data Cleaning**: Ask about cleaning results and applied corrections
    
    Please be more specific about what you'd like to know!
    """

def display_terminology_report(validation_result):
    """Display medical terminology validation results"""
    
    if "terminology_report" in validation_result:
        terminology_report = validation_result["terminology_report"]
        summary = validation_result.get("summary", {})
        
        # Display summary metrics - FIXED: Convert to int safely
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spelling Corrections", int(summary.get("corrections_count", 0)))
        with col2:
            st.metric("Standardizations", int(summary.get("standardizations_count", 0)))
        with col3:
            st.metric("Abbreviation Expansions", int(summary.get("expansions_count", 0)))
        with col4:
            st.metric("Total Issues Found", int(summary.get("total_issues", 0)))
        
        # Display detailed findings
        if terminology_report.get("corrections"):
            st.subheader("🔤 Spelling Corrections")
            corrections_df = pd.DataFrame([
                {"Original": orig, "Corrected": corr} 
                for orig, corr in terminology_report["corrections"].items()
            ])
            st.dataframe(corrections_df, use_container_width=True)
        
        if terminology_report.get("standardizations"):
            st.subheader("📋 Standardizations") 
            standardizations_df = pd.DataFrame([
                {"Original": orig, "Standardized": std} 
                for orig, std in terminology_report["standardizations"].items()
            ])
            st.dataframe(standardizations_df, use_container_width=True)
        
        if terminology_report.get("expansions"):
            st.subheader("🔍 Abbreviation Expansions")
            expansions_df = pd.DataFrame([
                {"Abbreviation": abbr, "Full Form": full} 
                for abbr, full in terminology_report["expansions"].items()
            ])
            st.dataframe(expansions_df, use_container_width=True)
    else:
        st.json(validation_result)

def display_cleaning_results(cleaning_result):
    """Display cleaning results in a user-friendly format"""
    
    quality_report = cleaning_result.get("quality_report", {})
    
    # FIXED: Convert all values to int for st.metric
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = quality_report.get('data_quality_score', 0)
        st.metric("Quality Score", f"{int(score)}/100")
    with col2:
        corrections = quality_report.get('medical_corrections_applied', 0)
        st.metric("Medical Corrections", int(corrections))
    with col3:
        missing_reduced = quality_report.get('missing_values_reduced', 0)
        st.metric("Missing Values Reduced", int(missing_reduced))
    with col4:
        duplicates_removed = quality_report.get('duplicates_removed', 0)
        st.metric("Duplicates Removed", int(duplicates_removed))
    
    # Display cleaning summary
    if 'cleaning_summary' in quality_report:
        st.subheader("🧹 Cleaning Operations Applied")
        for i, operation in enumerate(quality_report['cleaning_summary'], 1):
            st.text(f"{i}. {operation}")
    
    # Display before/after comparison - FIXED: Handle tuples properly
    if 'original_shape' in quality_report and 'cleaned_shape' in quality_report:
        st.subheader("📊 Before vs After")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Data:**")
            original_shape = quality_report['original_shape']
            if isinstance(original_shape, tuple):
                st.write(f"Shape: {original_shape[0]} rows × {original_shape[1]} columns")
            else:
                st.write(f"Shape: {original_shape}")
        with col2:
            st.write("**Cleaned Data:**")
            cleaned_shape = quality_report['cleaned_shape']
            if isinstance(cleaned_shape, tuple):
                st.write(f"Shape: {cleaned_shape[0]} rows × {cleaned_shape[1]} columns")
            else:
                st.write(f"Shape: {cleaned_shape}")

def main():
    st.title("🏥 GoML Multi-Agent Medical Data Cleaning System")
    st.markdown("**Simplified 3-Agent Architecture: Data Analysis → Medical Validation → Direct Cleaning**")
    st.markdown("---")
    
    # Initialize session state for queries and responses
    if 'user_queries' not in st.session_state:
        st.session_state.user_queries = []
    if 'agent_responses' not in st.session_state:
        st.session_state.agent_responses = []
    
    # Sidebar for file upload and controls
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your medical dataset", 
            type=['csv', 'xlsx'],
            help="Upload a CSV or Excel file containing medical data"
        )
        
        if uploaded_file:
            st.success("File uploaded successfully!")
            
        st.markdown("---")
        st.header("🤖 Agent Interaction")
        user_query = st.text_area(
            "Ask the agents:", 
            placeholder="Example: 'What medical terms were corrected?' or 'Show me the cleaning results'"
        )
        
        if st.button("💬 Send Query"):
            if user_query:
                # Get current data and results
                current_df = None
                current_results = None
                
                if uploaded_file:
                    try:
                        if uploaded_file.name.endswith('.csv'):
                            current_df = pd.read_csv(uploaded_file)
                        else:
                            current_df = pd.read_excel(uploaded_file)
                    except Exception as e:
                        st.error(f"Error loading file for query: {str(e)}")
                        current_df = None
                
                if 'results' in st.session_state:
                    current_results = st.session_state.results
                
                # Process the query with agents
                with st.spinner("🤖 Agents are processing your query..."):
                    agent_response = process_user_query(user_query, current_df, current_results)
                
                # Save query and response
                st.session_state.user_queries.append(user_query)
                st.session_state.agent_responses.append(agent_response)
                
                st.success("✅ Agents responded to your query!")
        
        # Display query history
        if st.session_state.user_queries:
            st.markdown("---")
            st.header("💬 Query History")
            
            for i, (query, response) in enumerate(zip(
                reversed(st.session_state.user_queries[-3:]),
                reversed(st.session_state.agent_responses[-3:])
            )):
                with st.expander(f"Query {len(st.session_state.user_queries)-i}: {query[:50]}..."):
                    st.markdown("**Your Question:**")
                    st.write(query)
                    st.markdown("**Agent Response:**")
                    st.markdown(response)
    
    # Main content area
    if uploaded_file:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Show data preview - FIXED: Proper int conversion
            with st.expander("📊 Original Data Preview", expanded=False):
                st.dataframe(df.head(20))
                
                # Basic statistics - FIXED: Extract int from tuples
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", int(df.shape[0]))
                with col2:
                    st.metric("Total Columns", int(df.shape[1]))
                with col3:
                    st.metric("Missing Values", int(df.isnull().sum().sum()))
                with col4:
                    st.metric("Duplicates", int(df.duplicated().sum()))
            
            # Process button
            if st.button("🚀 Start 3-Agent Processing Pipeline", type="primary"):
                with st.spinner("🤖 Running 3-Agent Pipeline: Analysis → Validation → Cleaning..."):
                    try:
                        orchestrator = MedicalDataOrchestrator()
                        results = orchestrator.process_data(df)
                        st.session_state.results = results
                        st.success("✅ 3-Agent Pipeline Complete! Your cleaned dataset is ready for download.")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        st.error("Please check your data format and try again.")
            
            # Display results if available
            if 'results' in st.session_state:
                results = st.session_state.results
                
                # Create tabs for the 3 agents
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Data Analysis", 
                    "🏥 Medical Validation", 
                    "🧹 Cleaning Results",
                    "📥 Download Clean Data"
                ])
                
                with tab1:
                    st.header("📊 Data Analysis Results")
                    st.json(results['analysis_result'])
                
                with tab2:
                    st.header("🏥 Medical Knowledge Validation")
                    validation_result = results['validation_result']['validation_result']
                    display_terminology_report(validation_result)
                
                with tab3:
                    st.header("🧹 Data Cleaning Results")
                    display_cleaning_results(results['cleaning_result'])
                
                with tab4:
                    st.header("📥 Download Cleaned Dataset")
                    
                    # Display cleaned data preview
                    cleaned_df = results['cleaned_dataframe']
                    st.subheader("🧽 Cleaned Dataset Preview")
                    st.success("✨ This dataset includes all medical terminology corrections and data quality improvements!")
                    st.dataframe(cleaned_df.head(20))
                    
                    # Comparison metrics - FIXED: Proper tuple handling
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Original Data")
                        st.metric("Rows", int(df.shape[0]))
                        st.metric("Missing Values", int(df.isnull().sum().sum()))
                        st.metric("Duplicates", int(df.duplicated().sum()))
                    
                    with col2:
                        st.subheader("🧽 Cleaned Data")
                        st.metric("Rows", int(cleaned_df.shape[0]))
                        st.metric("Missing Values", int(cleaned_df.isnull().sum().sum()))
                        st.metric("Duplicates", int(cleaned_df.duplicated().sum()))
                    
                    # Download button
                    csv = cleaned_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Complete Cleaned Dataset",
                        data=csv,
                        file_name="cleaned_medical_dataset_with_terminology_corrections.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
                    # Visualization comparison - FIXED: Proper tuple handling
                    if not cleaned_df.empty:
                        st.subheader("📈 Data Quality Improvement Visualization")
                        missing_orig = df.isnull().sum()
                        missing_clean = cleaned_df.isnull().sum()
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Original Missing Values", "After Cleaning"),
                            specs=[[{"type": "bar"}, {"type": "bar"}]]
                        )
                        
                        fig.add_trace(
                            go.Bar(x=missing_orig.index, y=missing_orig.values, name="Original"),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Bar(x=missing_clean.index, y=missing_clean.values, name="Cleaned"),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Agent processing log
                with st.expander("📝 3-Agent Processing Log", expanded=False):
                    if 'messages' in results and results['messages']:
                        for i, message in enumerate(results['messages']):
                            st.text(f"{i+1}. {message}")
                    else:
                        st.info("No processing messages available.")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Simplified 3-Agent Medical Data Cleaning System
        
        This streamlined system uses **3 specialized AI agents** for efficient medical data cleaning:
        
        - 🔍 **Data Analysis Agent**: Identifies data quality issues and patterns
        - 🏥 **Medical Knowledge Agent**: Validates and corrects medical terminology
        - 🧹 **Data Cleaning Agent**: Applies all corrections directly to your dataset
        
        ### Simple 3-Step Process:
        1. 📁 Upload your medical dataset (CSV or Excel)
        2. 🚀 Click "Start 3-Agent Processing Pipeline"
        3. 📥 Download your cleaned dataset with all corrections applied
        
        ### What Gets Cleaned:
        - ✅ Medical terminology corrections (typos, abbreviations, standardizations)
        - ✅ Missing value imputation
        - ✅ Duplicate removal
        - ✅ Outlier handling
        - ✅ Text standardization
        
        **Note**: Set your OpenAI API key in environment variables before starting.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🏥 **GoML August Hackathon** - Simplified 3-Agent Medical Data Cleaning System")

if __name__ == "__main__":
    main()
