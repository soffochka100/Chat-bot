import requests

API_KEY = "dfa9da3c88565da788036d8c26bcf817"  

def get_weather(city):
    """
    Получает погоду для указанного города через WeatherStack API
    """
    url = "http://api.weatherstack.com/current"
    params = {
        "access_key": API_KEY,
        "query": city,
        "units": "m",  
        "lang": "ru"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Ошибка API: {data['error']['info']}"

        temperature = data["current"]["temperature"]
        description = data["current"]["weather_descriptions"][0]
        wind_speed = data["current"]["wind_speed"]
        humidity = data["current"]["humidity"]
        feels_like = data["current"]["feelslike"]
        
        city_display = city.strip().title()

        weather_emoji = "" if "солнеч" in description.lower() else \
                       "" if "облач" in description.lower() else \
                       "" if "дожд" in description.lower() else \
                       "" if "снег" in description.lower() else \
                       ""

        return (f" Погода в городе {city_display}:\n"
                f"{weather_emoji} Сейчас: {description}\n"
                f" Температура: {temperature}°C (ощущается как {feels_like}°C)\n"
                f" Ветер: {wind_speed} м/с\n"
                f" Влажность: {humidity}%")

    except requests.exceptions.RequestException as e:
        return f" Ошибка подключения к сервису погоды: {e}"
    except (KeyError, IndexError) as e:
        return " Ошибка обработки данных о погоде"