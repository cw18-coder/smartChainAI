#!/usr/bin/env python3
"""
Test script for the smartChainAI MCP server forecasting tools
Tests run_arima and run_uvt with sample time series data
"""
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from datetime import datetime, timedelta

async def test_forecasting_tools():
    """Test the forecasting tools with sample data"""
    print("Testing smartChainAI MCP Server Forecasting Tools...")
    
    # Create transport for the containerized server
    transport = StreamableHttpTransport(url='http://localhost:8000/mcp')
    client = Client(transport)
    
    # Sample time series data - simple upward trend
    base_date = datetime(2025, 1, 1)
    train_data = []
    for i in range(30):  # 30 days of data
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        value = 100 + i * 2 + (i % 7) * 0.5  # Trend + weekly seasonality
        train_data.append({
            "unique_id": "series_1",
            "ds": date_str,
            "y": value
        })
    
    # Test data structure
    test_input = {
        "train_data": train_data,
        "freq": "D",  # Daily frequency
        "h": 7,       # Forecast 7 days ahead
        "level": [80, 95]  # Confidence levels
    }
    
    try:
        async with client:
            print("✅ Connected to MCP server")
            
            # Test run_arima
            print("\n🔮 Testing ARIMA forecasting:")
            print(f"  - Training data: {len(train_data)} observations")
            print(f"  - Forecast horizon: {test_input['h']} days")
            print(f"  - Confidence levels: {test_input['level']}")
            
            arima_result = await client.call_tool('run_arima', {"input": test_input})
            
            if arima_result.is_error:
                print(f"  ❌ ARIMA failed: {arima_result.content}")
            else:
                forecast_data = arima_result.data
                print(f"  ✅ ARIMA forecast generated: {len(forecast_data['data'])} predictions")
                print(f"  📊 Status: {forecast_data['status_code']} - {forecast_data['message']}")
                
                # Show first forecast point
                if forecast_data['data']:
                    first_forecast = forecast_data['data'][0]
                    print(f"  📈 First forecast: {first_forecast}")
            
            # Test run_uvt
            print("\n🎯 Testing Univariate Models forecasting:")
            
            uvt_result = await client.call_tool('run_uvt', {"input": test_input})
            
            if uvt_result.is_error:
                print(f"  ❌ UVT failed: {uvt_result.content}")
            else:
                forecast_data = uvt_result.data
                print(f"  ✅ UVT forecast generated: {len(forecast_data['data'])} predictions")
                print(f"  📊 Status: {forecast_data['status_code']} - {forecast_data['message']}")
                
                # Show first forecast point
                if forecast_data['data']:
                    first_forecast = forecast_data['data'][0]
                    print(f"  📈 First forecast: {first_forecast}")
            
            print("\n✅ All forecasting tests completed!")
            
    except Exception as e:
        print(f"❌ Error testing forecasting tools: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_explain_time_series_input():
    """Test the explanation tool"""
    print("\n📖 Testing time series input explanation:")
    
    transport = StreamableHttpTransport(url='http://localhost:8000/mcp')
    client = Client(transport)
    
    try:
        async with client:
            explanation = await client.call_tool('explain_time_series_input', {})
            print(f"  ✅ Got explanation ({len(str(explanation.data))} characters)")
            print(f"  📝 Preview: {str(explanation.data)[:200]}...")
            
    except Exception as e:
        print(f"  ❌ Error getting explanation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("smartChainAI MCP Server - Forecasting Tools Test")
    print("=" * 60)
    
    # Test explanation first
    asyncio.run(test_explain_time_series_input())
    
    # Test forecasting tools
    success = asyncio.run(test_forecasting_tools())
    
    print("=" * 60)
    if success:
        print("🎉 All forecasting tools are working correctly!")
    else:
        print("💥 Some forecasting tests failed!")
    print("=" * 60)