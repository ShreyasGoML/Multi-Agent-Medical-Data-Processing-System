from fastapi import APIRouter, HTTPException, UploadFile, File
from app.agent import execute_agent
import pandas as pd
import logging
from io import StringIO

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/process-csv")
async def analyze_csv_with_agent(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Read the uploaded file
        contents = await file.read()
        csv_string = contents.decode('utf-8')
        
        # Convert to DataFrame
        df = pd.read_csv(StringIO(csv_string))
        
        # Execute the medical agent
        agent = execute_agent()
        
        initial_state = {
            "csv_data": df,
            "filename": file.filename,
            "error": None,
            "analysis": None,
            "recommendations": None
        }
        
        result = agent.invoke(initial_state, config={"recursion_limit": 50})
        
        response = {
            "success": True,
            "filename": file.filename,
            "rows_processed": len(df),
            "columns": list(df.columns),
            "analysis": result.get("analysis"),
            "recommendations": result.get("recommendations"),
            "error": result.get("error")
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing CSV file: {str(e)}"
        )