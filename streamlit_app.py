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

def main():
    st.title("🏥 GoML Multi-Agent Medical Data Cleaning System")
    st.markdown("---")
    
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
            placeholder="Ask questions about your data or cleaning process..."
        )
        
        if st.button("💬 Send Query"):
            if user_query:
                st.session_state.user_queries = st.session_state.get('user_queries', [])
                st.session_state.user_queries.append(user_query)
                st.success("Query sent to agents!")
    
    # Main content area
    if uploaded_file:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Show data preview
            with st.expander("📊 Data Preview", expanded=False):
                st.dataframe(df.head(20))
                
                # Basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                with col2:
                    st.metric("Total Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                with col4:
                    st.metric("Duplicates", df.duplicated().sum())
            
            # Process button
            if st.button("🚀 Start Multi-Agent Processing", type="primary"):
                with st.spinner("🤖 Multi-Agent System Processing..."):
                    orchestrator = MedicalDataOrchestrator()
                    results = orchestrator.process_data(df)
                    st.session_state.results = results
                
                st.success("✅ Multi-Agent Processing Complete!")
            
            # Display results if available
            if 'results' in st.session_state:
                results = st.session_state.results
                
                # Create tabs for different result sections
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Data Analysis", 
                    "🏥 Medical Validation", 
                    "🧹 Cleaning Strategy", 
                    "💻 Generated Code", 
                    "✅ Quality Assurance",
                    "📈 Comparison Dashboard"
                ])
                
                with tab1:
                    st.header("📊 Data Analysis Results")
                    st.json(results['analysis_result'])
                
                with tab2:
                    st.header("🏥 Medical Knowledge Validation")
                    st.json(results['validation_result'])
                
                with tab3:
                    st.header("🧹 Cleaning Strategy")
                    st.json(results['cleaning_strategy'])
                
                with tab4:
                    st.header("💻 Generated Cleaning Code")
                    st.code(results['generated_code'], language='python')
                    
                    if st.button("📥 Download Code"):
                        st.download_button(
                            label="Download Python Code",
                            data=results['generated_code'],
                            file_name="medical_data_cleaning.py",
                            mime="text/python"
                        )
                
                with tab5:
                    st.header("✅ Quality Assurance Report")
                    st.json(results['qa_result'])
                    
                    # Display cleaned data
                    st.subheader("🧽 Cleaned Dataset")
                    st.dataframe(results['cleaned_dataframe'].head(20))
                    
                    if st.button("📥 Download Cleaned Data"):
                        csv = results['cleaned_dataframe'].to_csv(index=False)
                        st.download_button(
                            label="Download Cleaned CSV",
                            data=csv,
                            file_name="cleaned_medical_data.csv",
                            mime="text/csv"
                        )
                
                with tab6:
                    st.header("📈 Before vs After Comparison")
                    
                    # Comparison metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Original Data")
                        st.metric("Rows", df.shape[0])
                        st.metric("Missing Values", df.isnull().sum().sum())
                        st.metric("Duplicates", df.duplicated().sum())
                    
                    with col2:
                        st.subheader("🧽 Cleaned Data")
                        cleaned_df = results['cleaned_dataframe']
                        st.metric("Rows", cleaned_df.shape[0])
                        st.metric("Missing Values", cleaned_df.isnull().sum().sum())
                        st.metric("Duplicates", cleaned_df.duplicated().sum())
                    
                    # Visualization
                    if not cleaned_df.empty:
                        # Missing values comparison
                        missing_orig = df.isnull().sum()
                        missing_clean = cleaned_df.isnull().sum()
                        
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Original Missing Values", "Cleaned Missing Values"),
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
                
                # Agent messages
                st.header("📝 Agent Processing Log")
                for i, message in enumerate(results['messages']):
                    st.text(f"{i+1}. {message}")
        
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Multi-Agent Medical Data Cleaning System
        
        This system uses **5 specialized AI agents** orchestrated with **Langraph** to clean and validate medical data:
        
        - 🔍 **Data Analysis Agent**: Identifies data quality issues and inconsistencies
        - 🏥 **Medical Knowledge Agent**: Validates medical terminology and clinical data
        - 🧹 **Cleaning Strategy Agent**: Determines optimal cleaning approaches
        - 💻 **Code Generation Agent**: Creates executable Python code for data cleaning
        - ✅ **Quality Assurance Agent**: Validates cleaned data and ensures accuracy
        
        ### Instructions:
        1. 📁 Upload your medical dataset (CSV or Excel) using the sidebar
        2. 🚀 Click "Start Multi-Agent Processing" to begin
        3. 📊 Review results in the different tabs
        4. 💬 Interact with agents using the query box in the sidebar
        5. 📥 Download cleaned data and generated code
        
        **Note**: Make sure to set your OpenAI API key in the environment variables.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("🏥 **GoML August Hackathon** - Multi-Agent Medical Data Processing System")

if __name__ == "__main__":
    main()
