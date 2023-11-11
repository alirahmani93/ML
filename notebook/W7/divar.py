import json

import requests

base_url = 'https://api.divar.ir/v8/web-search/tehran/buy-residential'
all_posts: list = []
for i in range(1, 10):
    url = base_url + f'?page={i}'
    response = requests.get(url)
    data = json.loads(response.content.decode('utf-8'))
    post_list: list = data['web_widgets']['post_list']
    if len(post_list) != 0:
        all_posts.extend(post_list)
    else:
        print("Empty post list")
with open('divar.json', 'w', encoding='utf-8') as f:
    json.dump(all_posts, f, ensure_ascii=False, )

if "__main__" == __name__:
    print(len(all_posts))
    print(all_posts[:1])
