#!/usr/bin/env python3
"""
Test script for the smartChainAI MCP server
"""
import asyncio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

async def test_mcp_server():
    """Test the MCP server running in the Docker container"""
    print("Testing smartChainAI MCP Server...")
    
    # Create transport for the containerized server
    transport = StreamableHttpTransport(url='http://localhost:8000/mcp')
    client = Client(transport)
    
    try:
        async with client:
            print("✅ Connected to MCP server")
            
            # List available tools
            print("\n📋 Available tools:")
            tools = await client.list_tools()
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
            
            # Test health check
            print("\n🏥 Testing health check:")
            result = await client.call_tool('health_check', {})
            print(f"  Result: {result}")
            
            # Test explain_time_series_input
            print("\n📖 Time series input explanation:")
            explanation = await client.call_tool('explain_time_series_input', {})
            print(f"  Explanation preview: {str(explanation)[:200]}...")
            
            print("\n✅ All basic tests passed!")
            
    except Exception as e:
        print(f"❌ Error connecting to MCP server: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    if success:
        print("\n🎉 MCP Server is working correctly!")
    else:
        print("\n💥 MCP Server test failed!")