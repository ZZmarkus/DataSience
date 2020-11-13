import requests, json

url = ('http://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=ff0d334a96854958932475eb3d5e381a')

response = json.loads(requests.get(url).text)
print(response['totalResults'])

for author in response['articles']:
    print(author['author'])