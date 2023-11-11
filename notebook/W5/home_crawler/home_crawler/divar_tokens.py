import json
import pandas
import requests

r = requests.get('https://api.divar.ir/v8/web-search/tehran/buy-residential')
data = r.json().get('web_widgets').get('post_list')

all_home = []
for i in data:
    try:
        all_home.append(i['data']['action']['payload']['token'])
    except Exception as e:
        continue

pandas.Series(all_home).to_csv('divar_home_tokens.csv')

# all_home = []
# with open('divar_home_tokens.csv', 'a') as f:
#     for i in data:
#         try:
#             all_home.append(i['data']['action']['payload']['token'])
#         except Exception as e:
#             continue
#
#     json.dump(all_home, fp=f, )
# home = {
#     '': i['data']['image_count'],
#     '': i['data']['title'],
#     '': i['data']['top_description_text'],
#     '': i['data']['middle_description_text'],
#     '': i['data']['bottom_description_text'],
#     '': i['data']['action']['payload']['web_info']['title'],
#     '': i['data']['action']['payload']['web_info']['district_persian'],
#     '': i['data']['action']['payload']['web_info']['city_persian'],
#     '': i['data']['action']['payload']['web_info']['category_slug_persian'],
#     '': ,
# }