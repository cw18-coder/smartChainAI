# smartChainAI MCP Server

A FastMCP server providing time series forecasting capabilities through the Model Context Protocol (MCP).

## Features

- **ARIMA Forecasting** - Automated ARIMA model selection and forecasting
- **Univariate Models** - Multiple univariate forecasting models (AutoARIMA, CES, AutoETS, AutoTheta)
- **MSTL Decomposition** - Seasonal decomposition with univariate models
- **HTTP & STDIO Transport** - Flexible deployment options
- **Docker Support** - Containerized deployment ready
- **Confidence Intervals** - 80% and 95% prediction intervals

## Quick Start

### Using Docker (Recommended)

1. **Build the container:**
   ```bash
   docker build -t smartchain-ml .
   ```

2. **Run the server:**
   ```bash
   docker run -d -p 8000:8000 --name smartchain-ml-server smartchain-ml
   ```

3. **Test the server:**
   ```bash
   python tests/unit/test_mcp_client.py
   ```

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run with HTTP transport:**
   ```bash
   python src/ml_server.py --transport http
   ```

3. **Run with STDIO transport:**
   ```bash
   python src/ml_server.py --transport stdio
   ```

## Usage

The server provides several forecasting tools accessible via MCP:

### Available Tools

- `list_tools` - List all available tools
- `explain_time_series_input` - Get detailed input format documentation
- `run_arima` - Run ARIMA forecasting
- `run_uvt` - Run multiple univariate models
- `run_mstl_uvt` - Run univariate models with MSTL decomposition
- `health_check` - Server health status

### Input Format

All forecasting tools expect a `TimeSeriesInput` object:

```json
{
  "input": {
    "train_data": [
      {"unique_id": "series_1", "ds": "2025-01-01", "y": 100.0},
      {"unique_id": "series_1", "ds": "2025-01-02", "y": 102.0}
    ],
    "freq": "D",
    "h": 7,
    "level": [80, 95]
  }
}
```

### Example with FastMCP Client

```python
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def forecast_example():
    transport = StreamableHttpTransport(url='http://localhost:8000/mcp')
    client = Client(transport)
    
    test_data = {
        "input": {
            "train_data": [
                {"unique_id": "test", "ds": "2025-01-01", "y": 100},
                {"unique_id": "test", "ds": "2025-01-02", "y": 102},
                # ... more data points
            ],
            "freq": "D",
            "h": 3,
            "level": [80, 95]
        }
    }
    
    async with client:
        result = await client.call_tool('run_arima', test_data)
        print(result.data)

asyncio.run(forecast_example())
```

## API Endpoints

When running in HTTP mode, the server is accessible at:
- Base URL: `http://localhost:8000/mcp/`
- Health Check: Available via `health_check` tool
- All MCP communication follows JSON-RPC 2.0 protocol

## Project Structure

```
smartChainAI/
├── src/
│   ├── ml_server.py          # Main MCP server
│   ├── convert_ts_data.py    # Data conversion utilities
│   └── models/
│       ├── tsutils.py        # Time series utilities
│       └── univariate.py     # Univariate forecasting models
├── tests/
│   └── unit/
│       ├── test_mcp_client.py           # MCP client tests
│       ├── test_simple_arima.py         # ARIMA tests
│       ├── test_simple_uvt.py           # UVT tests
│       └── test_forecasting_tools.py    # Comprehensive tests
├── data/                     # Sample data files
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Docker Compose setup
└── requirements.txt         # Python dependencies
```

## Development

### Running Tests

```bash
# Test basic MCP connectivity
python tests/unit/test_mcp_client.py

# Test ARIMA forecasting
python tests/unit/test_simple_arima.py

# Test univariate models
python tests/unit/test_simple_uvt.py

# Run comprehensive forecasting tests
python tests/unit/test_forecasting_tools.py
```

### Adding New Models

1. Implement your model in `src/models/`
2. Add a new `@mcp.tool()` method in `ml_server.py`
3. Follow the `TimeSeriesInput` → `APIResponse` pattern
4. Add tests in `tests/unit/`

## Dependencies

- **FastMCP** - Model Context Protocol server framework
- **StatsForecast** - Time series forecasting models
- **Pandas** - Data manipulation
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server for HTTP transport

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Check existing tests for usage examples
- Use `explain_time_series_input` tool for input format help
- Review server logs for debugging information