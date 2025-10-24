from fastmcp import FastMCP
import json
import requests

mcp = FastMCP("Weather and Wikipedia Assistant")


@mcp.tool()
def get_weather(latitude: str, longitude: str) -> str:
    """
    Get the current weather in a given location

    Args:
        latitude: Latitude of the location for which to get the weather
        longitude: Longitude of the location for which to get the weather

    Returns:
        JSON with current weather information
    """
    try:

        weather_response = requests.get(
            url="https://api.open-meteo.com/v1/forecast",
            params={"latitude": latitude, "longitude": longitude, "current_weather": True},
            verify=False,
            proxies={'http': '', 'https': ''},
        )
        weather_info = weather_response.json()["current_weather"]

        return json.dumps({
            "success": True,
            "location": {"latitude": latitude, "longitude": longitude},
            "weather": weather_info
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Weather lookup error: {str(e)}"
        })


@mcp.tool()
def facts_from_wikipedia(topic: str) -> str:
    """
    Get Wikipedia facts about a given topic, location, person etc.

    Args:
        topic: A topic, location, person etc. to search for on Wikipedia

    Returns:
        JSON with Wikipedia information
    """
    try:
        headers = {
            'User-Agent': 'MCPBot/1.0 (Educational purpose only)',
            'Accept': 'application/json'
        }

        wikipedia_response = requests.get(
            url=f"https://en.wikipedia.org/w/api.php?action=query&prop=revisions&titles={topic}&rvprop=content&format=json",
            headers=headers,
            verify=False,
            proxies={'http': '', 'https': ''},
        )
        wiki_info = wikipedia_response.json()

        return json.dumps({
            "success": True,
            "topic": topic,
            "data": wiki_info
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Wikipedia lookup error: {str(e)}"
        })


if __name__ == "__main__":
    # Run MCP server in stdio mode
    mcp.run(show_banner=False)
