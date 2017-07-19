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
        time.sleep(5)
        print(url)
        
        FLAG = False
        urls = []   
        
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
                break
            elif FLAG == True and tag != after:
                tmp = tag.next_element.next_element.next_element
                
                if tmp != '\n' and tmp != after.string:
                   #print(tmp['src'])
                   urls.append(tmp['src'])
    
        save_name = os.path.join('csv_files', urls[0].split('/')[-1][:-6] + '.csv')
        sub_dir_name = urls[0].split('/')[-1][:-6]
        

        with open(save_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(urls)

        if not os.path.exists(os.path.join(dir_name, sub_dir_name)):
            os.mkdir(os.path.join(dir_name, sub_dir_name))
        
        for url in urls:
           
            img_save_name = url.split('/')[-1]
            
            if url.split('.')[-1] not in ('jpg', 'png', 'jpeg'):
                continue
            path = os.path.join(dir_name, sub_dir_name, img_save_name)
            r = requests.get(url)

            if r.status_code == 200:            
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        f.write(chunk)


def get_urls(url):
    lists = []
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    main_div = soup.find('div', {'class': 'post_content'})
    table = main_div.find('table')
    urls = table.findAll('a')
    
    for u in urls:
        lists.append(u['href'])

    with open(os.path.join('csv_files', 'name_list.csv'), 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(lists)
    
    return lists

if __name__ == "__main__":

    #lists = get_urls(os.path.join(BASE_URL, category[0]))
    #download(lists)
    lists = get_urls(os.path.join(BASE_URL, category[1]))
    download(lists)
    lists = get_urls(os.path.join(BASE_URL, category[1]))
