import requests
from bs4 import BeautifulSoup
import re
import os
import random
import time
import sys
from queue import Queue
from datetime import datetime, timedelta

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'referer': 'https://www.google.com/'}

#try to get the content of the page with the different random user agent
user_agent_list = [
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0; Acoo Browser; SLCC1; .NET CLR 2.0.50727; Media Center PC 5.0; .NET CLR 3.0.04506)",
    "Mozilla/4.0 (compatible; MSIE 7.0; AOL 9.5; AOLBuild 4337.35; Windows NT 5.1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"   
]

def get_html(url):
    try:
        headers['User-Agent'] = random.choice(user_agent_list)
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except Exception as e:
        print(e)
        return ""


def crawler_ch():
    delta = timedelta(days=1)
    start_date = datetime.today().date() - delta
    end_date = datetime(2021, 1, 1).date()
    parse_day = start_date
    timeline = (start_date - end_date).days
    base_url = 'http://paper.people.com.cn/rmrb/html/'
    while timeline >= 0:
        url = base_url + parse_day.isoformat()[:-3] + "/" + parse_day.isoformat()[8:10] + "/nbs.D110000renmrb_01.htm"
        html = get_html(url)
        print(f"crawling {url}")
        soup = BeautifulSoup(html, 'html.parser')
        next_pages = [a['href'] for a in soup.select("div.swiper-container div.swiper-slide a")]
        page_links = base_url + parse_day.isoformat()[:-3] + "/" + parse_day.isoformat()[8:10] + "/"

        for next_page in next_pages:
            page_link = page_links + next_page
            html = get_html(page_link)
            soup = BeautifulSoup(html, 'html.parser')
            next_passages = [a['href'] for a in soup.select("ul.news-list li a")]

            for next_passage in next_passages:
                next_passage_url = page_links + next_passage
                html = get_html(next_passage_url)
                soup = BeautifulSoup(html, 'html.parser')

                for article in soup.select('div.article'):
                    title = article.select_one('h1').get_text(strip=True)
                    author = article.select_one('p.sec').get_text(strip=True)
                    contents = [p.get_text(strip=True) for p in article.select('div#ozoom p')]
                    print(f"saving {title}")
                    with open('chinese.txt', 'a', encoding='utf-8') as f:
                        f.write(title + '\n' + author + '\n' + '\n'.join(contents) + '\n\n')

        parse_day -= delta
        timeline -= 1
        print(f"the rest of the days: {timeline}")

def crawler_en():
    base_url_en = "https://www.chinadaily.com.cn"
    titles = ['china','world','business','lifestyle','sports','opinion','culture','regional','travel']
    html = get_html(base_url_en)
    soup = BeautifulSoup(html, 'html.parser')
    q = Queue()
    q.put(base_url_en)
    pattern = r'^https://www\.chinadaily\.com\.cn/a/\d{6}/\d{2}/WS.+\.html$'
    for title in titles:
        q.put(base_url_en + '/' + title)
    while not q.empty():    
        url = q.get()
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        links =  []

        for link in soup.find_all('a'):
            href = link.get('href')
            print(href)
            if href:
                if href.startswith('//'):
                    href = 'https:' + href
                if re.match(pattern, href):
                    links.append(href)
                
        for link in links:
            print(f"crawling {link}")
            html = get_html(link)
            soup = BeautifulSoup(html, 'html.parser')
            text = ""
            for div in soup.find_all('div',id = 'Content'):
                if div:
                    text += div.get_text()
            with open('./homework1/english.txt', 'a', encoding='utf-8') as f:
                f.write(text)

def crawler_en_xinhua():
    base_url = "https://english.news.cn"
    total_size = 0
    max_size = 1024 * 1024 * 20
    q = Queue()
    q.put((base_url, 0))
    pattern = r'^https://english\.news\.cn/\d{8}/.+c\.html$'
    pattern_prefix = r'^https://english\.news\.cn/.+'
    while not q.empty():
        url, depth = q.get()
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        pre_url = None
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and depth < 5:
                if href.startswith('/'):
                    href = base_url + href
                elif href.startswith('#'):
                    continue
                if re.match(pattern, href):
                    if href == pre_url:
                        continue
                    pre_url = href
                    content = extract_xinhua(href)
                    total_size += sys.getsizeof(content)
                    save_text(content, 'english.txt')
                    if total_size > max_size:
                        break

                elif re.match(pattern_prefix, href):
                    q.put((href, depth + 1))
                

def extract_xinhua(link):
    print(f"crawling {link}")
    html = get_html(link)
    soup = BeautifulSoup(html, 'html.parser')
    text = ""
    for div in soup.find_all('div',class_ = 'content'):
        if div:
            text += div.get_text()
    return text

def save_text(text, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(text)



if __name__ == '__main__':
    # crawler_ch()
    # crawler_en()
    crawler_en_xinhua()