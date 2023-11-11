import requests
from bs4 import BeautifulSoup

r = requests.get('https://bootcamp.mapsahr.com/bootcamps/', headers={
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'})

bs = BeautifulSoup(r.text, 'lxml')
camp_data = bs.findAll('div','jet-listing-grid__item')

for i in camp_data:
    a = i.findAll('a','jet-listing-dynamic-link__link')
    print(a)