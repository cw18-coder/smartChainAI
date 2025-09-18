from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast
from models.univariate import BaseUnivariateForecaster, MSTLUnivariateForecaster
import pandas as pd
import uvicorn
import argparse
import json
import traceback
from typing import Any, Union

# Initialize FastMCP
mcp = FastMCP(
    name="smartChainAI_ML_Server",
    instructions="This is a machine learning server that provides time series forecasting using the ARIMA model./" \
    "You can list available tools and run the ARIMA model on provided time series data. More tools will be added in the future.",
)

class DataRecord(BaseModel):
    unique_id: int | str
    ds: str
    y: float | int

class TimeSeriesInput(BaseModel):
    # Use a boolean for strict mode (not a string) to avoid Pydantic SchemaError
    model_config = ConfigDict(strict=True)

    # Accept a list of well-typed records so JSON validation is predictable
    train_data: list[DataRecord]
    test_data: list[DataRecord] | None = None
    freq: str
    h: int

    # Accept numeric levels as floats (integers will be coerced to floats)
    level: list[float] | None = None

    id_col: str = "unique_id"
    time_col: str = "ds"
    target_col: str = "y"

class APIResponse(BaseModel):
    """Standardized API response format"""
    status_code: int
    message: str
    data: Any = None
    error: str | None = None

def prepare_dataframe_from_input(input: TimeSeriesInput) -> pd.DataFrame | dict:
    """
    Common function to convert TimeSeriesInput to a prepared DataFrame.
    
    Returns:
        pd.DataFrame: Prepared DataFrame with datetime conversion
        dict: Error dict if validation fails
    """
    # Convert Pydantic records (or plain dicts) into a list of dictionaries
    try:
        data_records = [r.model_dump() if hasattr(r, "model_dump") else r for r in input.train_data]
    except Exception:
        data_records = input.train_data

    # Validate that parsed data is a list
    if not isinstance(data_records, list) or len(data_records) == 0:
        return {"error": "Parsed train_data must be a non-empty list of records"}

    # Convert list of records to DataFrame
    df = pd.DataFrame(data_records)

    # Ensure required columns exist
    required_cols = {input.id_col, input.time_col, input.target_col}
    missing = required_cols - set(df.columns)
    if missing:
        return {"error": f"Missing required fields in data records: {', '.join(missing)}"}

    # convert time column to datetime
    df[input.time_col] = pd.to_datetime(df[input.time_col])

    return df

# Define the tools
@mcp.tool()
async def list_tools():
    """List available tools."""
    return [
        "list_tools",
        "explain_time_series_input",
        "run_arima",
        "run_uvt",
        "run_mstl_uvt",
        "health_check",
        ]

@mcp.tool()
async def explain_time_series_input():
    """Explain the time series input format."""
    explanation = (
        "The server accepts time series input in JSON format matching the `TimeSeriesInput` model:\n\n"
        "Required fields:\n"
        "   - 'train_data': list of records where each record is an object with:\n"
        "       - 'unique_id' (int or str) — the series identifier\n"
        "       - 'ds' (string) — ISO date for the observation\n"
        "       - 'y' (number) — the target value\n"
        "   - 'freq': string frequency (e.g., 'D' for daily)\n"
        "   - 'h': integer forecast horizon\n\n"
        "Optional fields:\n"
        "   - 'test_data': optional list of records with same structure as train_data for validation\n"
        "   - 'level': list of confidence levels expressed as percentages (e.g., [80, 95]) or floats (e.g., [0.8, 0.95]) — percentages are recommended\n"
        "   - 'id_col', 'time_col', 'target_col': names of the respective fields (defaults: 'unique_id', 'ds', 'y')\n\n"
        "   Example JSON payload:\n"
        "   {\n"
        "       \"train_data\": [{\"unique_id\": 1, \"ds\": \"2025-09-01\", \"y\": 100.0}, {\"unique_id\": 1, \"ds\": \"2025-09-02\", \"y\": 101.0}],\n"
        "       \"test_data\": [{\"unique_id\": 1, \"ds\": \"2025-09-03\", \"y\": 102.0}],\n"
        "       \"freq\": \"D\",\n"
        "       \"h\": 7,\n"
        "       \"level\": [80, 95]\n"
        "   }\n\n"
        "   You can validate this JSON locally using the `TimeSeriesInput` model: `TimeSeriesInput.model_validate_json(json_string)`.\n\n"
        "Notes:\n"
        "- The train_data is used for model training and forecasting\n"
        "- The test_data is optional and can be used for model validation (currently not utilized in ARIMA implementation)\n"
        "- Prefer percentage-style `level` values (e.g., [80,95]) to match what the forecasting code expects."
    )
    return explanation

@mcp.tool()
async def run_arima(input: TimeSeriesInput) -> dict:
    """Run ARIMA model on the provided TimeSeriesInput payload.

    This accepts a `TimeSeriesInput` Pydantic model (or equivalent JSON) with fields:
      - train_data: list of records (each record must contain the id_col, time_col, and target_col)
      - test_data: optional list of records with same structure as train_data
      - freq: frequency string (e.g., 'D')
      - h: forecast horizon (int)
      - level: optional list of confidence levels (percentages preferred, e.g., [80,95])
      - id_col/time_col/target_col: optional column names (defaults: 'unique_id','ds','y')

    The function converts the incoming records to a DataFrame, coerces the time column to datetime, calls StatsForecast with AutoARIMA, and returns the forecast as a structured response.
    """
    try:
        # Prepare DataFrame using the common function
        df = prepare_dataframe_from_input(input)
        
        # Check if an error occurred during DataFrame preparation
        if isinstance(df, dict) and "error" in df:
            return APIResponse(
                status_code=400,
                message="Data preparation failed",
                error=df["error"]
            ).model_dump()

        # Default levels (percentages) and normalize if user passed probabilities (0-1)
        levels = input.level if input.level is not None else [80, 95]
        try:
            if any(0 < lvl <= 1 for lvl in levels):
                levels = [float(lvl) * 100 for lvl in levels]
        except Exception as e:
            return APIResponse(
                status_code=400,
                message="Invalid confidence levels",
                error=f"Invalid 'level' values; expected list of numbers: {str(e)}"
            ).model_dump()

        # Initialize and fit the ARIMA model
        model = StatsForecast(models=[AutoARIMA()], freq=input.freq)

        # Generate forecasts
        forecast = model.forecast(
            df=df,
            h=input.h,
            level=levels,
        )

        forecast_data = forecast.to_dict(orient="records")
        
        return APIResponse(
            status_code=200,
            message=f"ARIMA forecast generated successfully for {len(forecast_data)} predictions",
            data=forecast_data
        ).model_dump()

    except Exception as e:
        # Capture full stack trace for debugging
        error_stack = traceback.format_exc()
        return APIResponse(
            status_code=500,
            message="ARIMA forecasting failed",
            error=error_stack
        ).model_dump()
    
@mcp.tool()
async def run_uvt(input: TimeSeriesInput) -> dict:
    """Run multiple univariate models on the provided TimeSeriesInput payload.

    This accepts a `TimeSeriesInput` Pydantic model (or equivalent JSON) with fields:
      - train_data: list of records (each record must contain the id_col, time_col, and target_col)
      - test_data: optional list of records with same structure as train_data
      - freq: frequency string (e.g., 'D')
      - h: forecast horizon (int)
      - level: optional list of confidence levels (percentages preferred, e.g., [80,95])
      - id_col/time_col/target_col: optional column names (defaults: 'unique_id','ds','y')

    The function converts the incoming records to a DataFrame, coerces the time column to datetime, calls StatsForecast with multiple univariate models, and returns the forecast as a structured response.
    """
    try:
        # Prepare DataFrame using the common function
        df = prepare_dataframe_from_input(input)
        
        # Check if an error occurred during DataFrame preparation
        if isinstance(df, dict) and "error" in df:
            return APIResponse(
                status_code=400,
                message="Data preparation failed",
                error=df["error"]
            ).model_dump()

        # Default levels (percentages) and normalize if user passed probabilities (0-1)
        levels = input.level if input.level is not None else [80, 95]
        try:
            if any(0 < lvl <= 1 for lvl in levels):
                levels = [float(lvl) * 100 for lvl in levels]
        except Exception as e:
            return APIResponse(
                status_code=400,
                message="Invalid confidence levels",
                error=f"Invalid 'level' values; expected list of numbers: {str(e)}"
            ).model_dump()

        # Initialize the BaseUnivariateForecaster
        forecaster = BaseUnivariateForecaster(
            freq=input.freq,
            h=input.h,
            level=levels,
        )

        # Generate forecasts using the forecaster
        forecasts = forecaster.predict(
            df=df,
            id_col=input.id_col,
            time_col=input.time_col,
            target_col=input.target_col,
        )

        return APIResponse(
            status_code=200,
            message=f"Univariate forecasts generated successfully for {len(forecasts)} predictions",
            data=forecasts
        ).model_dump()
    except Exception as e:
        # Capture full stack trace for debugging
        error_stack = traceback.format_exc()
        return APIResponse(
            status_code=500,
            message="Univariate forecasting failed",
            error=error_stack
        ).model_dump()
    
@mcp.tool()
async def run_mstl_uvt(input: TimeSeriesInput) -> dict:
    """Run multiple univariate models with MSTL decomposition on the provided TimeSeriesInput payload.

    This accepts a `TimeSeriesInput` Pydantic model (or equivalent JSON) with fields:
      - train_data: list of records (each record must contain the id_col, time_col, and target_col)
      - test_data: optional list of records with same structure as train_data
      - freq: frequency string (e.g., 'D')
      - h: forecast horizon (int)
      - level: optional list of confidence levels (percentages preferred, e.g., [80,95])
      - id_col/time_col/target_col: optional column names (defaults: 'unique_id','ds','y')

    The function converts the incoming records to a DataFrame, coerces the time column to datetime, calls StatsForecast with multiple univariate models wrapped in MSTL, and returns the forecast as a structured response.
    """
    try:
        # Prepare DataFrame using the common function
        df = prepare_dataframe_from_input(input)
        
        # Check if an error occurred during DataFrame preparation
        if isinstance(df, dict) and "error" in df:
            return APIResponse(
                status_code=400,
                message="Data preparation failed",
                error=df["error"]
            ).model_dump()

        # Determine minimum time series length for seasonality selection
        try:
            min_ts_len = df.groupby(input.id_col).size().min()
            if min_ts_len < 2:
                return APIResponse(
                    status_code=400,
                    message="Insufficient data for forecasting",
                    error=f"Minimum time series length is {min_ts_len}, but at least 2 observations are required"
                ).model_dump()
        except Exception as e:
            return APIResponse(
                status_code=400,
                message="Failed to analyze time series data",
                error=f"Error calculating minimum time series length: {str(e)}"
            ).model_dump()

        # Initialize the MSTLUnivariateForecaster
        try:
            forecaster = MSTLUnivariateForecaster(
                min_ts_len=min_ts_len,
                freq=input.freq,
                h=input.h,
                level=input.level,
            )
        except Exception as e:
            return APIResponse(
                status_code=500,
                message="Failed to initialize forecaster",
                error=f"MSTLUnivariateForecaster initialization error: {str(e)}"
            ).model_dump()

        # Generate forecasts using the forecaster
        forecasts = forecaster.predict(
            df=df,
            id_col=input.id_col,
            time_col=input.time_col,
            target_col=input.target_col,
        )

        return APIResponse(
            status_code=200,
            message=f"Univariate forecasts generated successfully for {len(forecasts)} predictions using MSTL decomposition",
            data=forecasts
        ).model_dump()

    except Exception as e:
        # Capture full stack trace for debugging
        error_stack = traceback.format_exc()
        return APIResponse(
            status_code=500,
            message="Univariate forecasting failed",
            error=error_stack
        ).model_dump()

@mcp.tool()
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastMCP server.")
    parser.add_argument(
        "--transport", 
        choices=["http", "stdio"], 
        default="http", 
        help="Choose the transport method for the FastMCP server."
    )
    args = parser.parse_args()

    if args.transport == "http":
        app = mcp.http_app()
        # run with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    elif args.transport == "stdio":
        mcp.run(transport="stdio")