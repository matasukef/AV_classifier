import os
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import argparse
import sys
import pandas as pd
import time

BASE_URL="http://sumomo-ch.com/"

# save dir
dir_name = os.path.join('..', 'Images', 'add')


def download(lists, name):
    if not os.path.exists(os.path.join(dir_name, name)):
        os.mkdir(os.path.join(dir_name, name))

    for url in tqdm(lists):
        if url.split('.')[-1] not in ('jpg', 'png', 'jpeg'):
            continue
        path = os.path.join(dir_name, name, url.split('/')[-1])
        r = requests.get(url)

        if r.status_code == 200:            
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)


def get_urls(blog_no):
    lists = []
    url = BASE_URL + "blog-entry-" + str(blog_no) + ".html"
    print(url)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    main_div = soup.find('div', {'id': 'contain'})
    left_div = main_div.find('div', {'id': 'left'})
    urls = left_div.findAll('a')
    
    for u in urls:
        if u.string == None and u.has_attr('target') and not u.has_attr('title'):
            lists.append(u['href'])

    return lists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="download images from somomo")
    parser.add_argument(
        'blog_no', type=str
    )
    parser.add_argument(
        'save_dir', type=str
    )
    args = parser.parse_args()

    lists = get_urls(args.blog_no)
    download(lists, args.save_dir)
