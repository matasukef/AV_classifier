import os
import csv
from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import argparse
import sys
import pandas as pd
import time

BASE_URL="http://eropalace21.com/av-gazou"

category = ['oppai', 'bishojo']


# save dir

def download(lists):
    
    dir_name = os.path.join('..', 'Images', 'eropalace21')

    for url in tqdm(lists):
        
        urls = []   
        FLAG = True
        
        if not os.path.exists(os.path.join(dir_name, name)):
            os.mkdir(os.path.join(dir_name, name))
        
        r = requests.get(url)
        soup = BeautifulSoup(r.content, 'lxml')
        main_div = soup.find('div', {'class':'post_content'})
        sep = main_div.find_all('h3')
        
        before = sep[0]
        after = sep[1]

        for tag in main_div:
            if FLAG == False and tag == before:
                FLAG = True
            elif FLAG == True and tag == after:
                FALG = False
            elif FLAG == True and tag != after:
                   urls.append(tag['href'])
    
        save_name = urls[0].split('/')[-1][-5] + '.csv'
        with open(save_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(urls)

        for url in urls:
            if url.split('.')[-1] not in ('jpg', 'png', 'jpeg'):
                continue
            path = os.path.join(dir_name, save_name, url.split('/')[-1])
            r = requests.get(url)

            if r.status_code == 200:            
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)


def get_urls():
    lists = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    main_div = soup.find('div', {'class': 'post_content'})
    table = main_div.find('table')
    urls = table.findAll('a')
    
    for u in urls:
        lists.append(u['href'])

    with open('name_list.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(lists)
    
    return lists

if __name__ == "__main__":

    lists = get_urls(os.path.join(BASE_URL, category[0]))
    download(lists)