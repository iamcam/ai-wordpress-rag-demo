import base64
import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

wordpress_user = os.getenv('WORDPRESS_APPLICATION_USER')
wordpress_password = os.getenv('WORDPRESS_APPLICATION_PASS')
wordpress_domain = os.getenv('WORDPRESS_DOMAIN')
wp_root_path = os.getenv('WP_ROOT_PATH', "wp-json/wp/v2")

wordpress_credentials = wordpress_user + ":" + wordpress_password
wordpress_token = base64.b64encode(wordpress_credentials.encode())
wordpress_header = {'Authorization': 'Basic ' + wordpress_token.decode('utf-8')}


def read_wordpress_posts(limit=100):
    api_url = f'https://{wordpress_domain}/{wp_root_path}/posts?per_page={int(limit)}&_fields=id,title.rendered,content,link,categories,tags'
    response = requests.get(api_url)
    response_json = response.json()
    return response_json


def get_total_pagecount():
    api_url = f"https://{wordpress_domain}/{wp_root_path}/posts?page=1&per_page=100"
    response = requests.get(api_url)
    pages_count = response.headers['X-WP-TotalPages']
    return int(pages_count)

def read_wordpress_post_with_pagination():
    total_pages = get_total_pagecount()
    current_page = 1
    all_page_items_json = []
    while current_page <= total_pages:
        api_url = f"https://{wordpress_domain}/{wp_root_path}/posts?page={current_page}&per_page=100&_fields=id,title.rendered,content,link,categories,tags"
        page_items = requests.get(api_url)
        page_items_json = page_items = page_items.json()
        all_page_items_json.extend(page_items_json)
        current_page = current_page + 1
    return all_page_items_json


# entity_type = [ posts | tags | categories]
def read_wordpress_entity(entity_type, limit=100, extra_params=""):
    api_url = f'https://{wordpress_domain}/{wp_root_path}/{entity_type}?per_page={int(limit)}&{extra_params}'
    response = requests.get(api_url)
    response_json = response.json()
    return response_json



tags = read_wordpress_entity('tags',extra_params="_fields=id,name,slug")
categories = read_wordpress_entity('categories',extra_params="_fields=id,name,slug")
posts = read_wordpress_post_with_pagination()


if not os.path.exists('data'):
    os.makedirs('data')
with open('data/posts.json', 'w') as outfile:
    json.dump(posts, outfile, indent=4)
