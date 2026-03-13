import requests

API_KEY = ""
BASE_URL = "http://api.weatherstack.com/current"

def get_weather(city):
    params = {
        'access_key': API_KEY,
        'query': city,
        'units': 'm'
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if not data.get('current'):
            return "Не удалось получить данные о погоде"
        
        current = data['current']
        location = data['location']['name']
        country = data['location']['country']
        temp = current['temperature']
        feels_like = current['feelslike']
        desc = current['weather_descriptions'][0]
        wind_speed = current['wind_speed']
        humidity = current['humidity']
        
        return (f"Погода в {location}, {country}:\n"
                f"Температура: {temp}°C (ощущается как {feels_like}°C)\n"
                f"Описание: {desc}\n"
                f"Ветер: {wind_speed} м/с\n"
                f"Влажность: {humidity}%")
                
    except Exception as e:
        return f"Ошибка: {e}"