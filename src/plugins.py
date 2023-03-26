import requests
import os
import datetime


def get_weather(city, units="metric"):
    """Get weather data for a given city.

    Args:
        city (str): The city to get weather data for.
        units (str): The units to use. Can be "metric" or "imperial".

    Returns:
        dict: A dictionary containing the weather data.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    api_key = os.environ["OPENWEATHER_API_KEY"]

    def get_lat_long(city):
        """Get the latitude and longitude for a given city.

        Args:
            city (str): The city to get the latitude and longitude for.

        Returns:
            tuple: A tuple containing the latitude and longitude.
        """
        base_url = "http://api.openweathermap.org/geo/1.0/direct?"

        api_key = os.environ["OPENWEATHER_API_KEY"]

        url = f"{base_url}appid={api_key}&q={city}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            return data[0]["lat"], data[0]["lon"]
        else:
            return None

    lat, lon = get_lat_long(city)

    url = f"{base_url}appid={api_key}&lat={lat}&lon={lon}&units={units}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        main = data["main"]
        wind = data["wind"]
        weather_desc = data["weather"][0]["description"]

        return {
            "city": city,
            "temperature": main["temp"],
            "humidity": main["humidity"],
            "pressure": main["pressure"],
            "wind_speed": wind["speed"],
            "weather_description": weather_desc,
        }
    else:
        return None


def get_time():
    """Get the current time.

    Returns:
        str: The current time."""
    now = datetime.datetime.now()
    return now.strftime("%H:%M")
