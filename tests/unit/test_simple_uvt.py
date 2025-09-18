#!/usr/bin/env python3
"""
Simple test for UVT (Univariate) forecasting tool
"""
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def test_uvt():
    """Test UVT with simple data"""
    
    # Simple test data - 10 points with trend
    train_data = [
        {"unique_id": "test", "ds": "2025-01-01", "y": 100},
        {"unique_id": "test", "ds": "2025-01-02", "y": 102},
        {"unique_id": "test", "ds": "2025-01-03", "y": 104},
        {"unique_id": "test", "ds": "2025-01-04", "y": 106},
        {"unique_id": "test", "ds": "2025-01-05", "y": 108},
        {"unique_id": "test", "ds": "2025-01-06", "y": 110},
        {"unique_id": "test", "ds": "2025-01-07", "y": 112},
        {"unique_id": "test", "ds": "2025-01-08", "y": 114},
        {"unique_id": "test", "ds": "2025-01-09", "y": 116},
        {"unique_id": "test", "ds": "2025-01-10", "y": 118}
    ]
    
    test_input = {
        "train_data": train_data,
        "freq": "D",
        "h": 3,  # Forecast 3 days
        "level": [80, 95]
    }
    
    transport = StreamableHttpTransport(url='http://localhost:8000/mcp')
    client = Client(transport)
    
    async with client:
        print("Testing UVT (Univariate Models)...")
        result = await client.call_tool('run_uvt', {"input": test_input})
        
        if result.is_error:
            print(f"Error: {result.content}")
        else:
            print("Success!")
            print(f"Data: {result.data}")

if __name__ == "__main__":
    asyncio.run(test_uvt())