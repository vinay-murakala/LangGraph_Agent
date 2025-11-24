import os
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """
    Fetches the current weather for a given city name.
    Args:
        city: The name of the city (e.g., "London", "New York").
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")

    if not api_key:
        return "Error: API key missing."
    if not city:
        return "Error: No city specified."

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)

        if response.status_code != 200:
            return f"Error: Provider returned status {response.status_code}: {response.text}"

        data = response.json()
        
        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        
        return f"The weather in {city} is currently {weather_desc} with a temperature of {temp}°C and {humidity}% humidity."
        
    except Exception as e:
        return f"Failed to fetch weather: {str(e)}"
