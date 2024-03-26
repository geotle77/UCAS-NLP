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
    "Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Win64; x64; Trident/5.0; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 2.0.50727; Media Center PC 6.0)",
    "Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; .NET CLR 1.0.3705; .NET CLR 1.1.4322)",
    "Mozilla/4.0 (compatible; MSIE 7.0b; Windows NT 5.2; .NET CLR 1.1.4322; .NET CLR 2.0.50727; InfoPath.2; .NET CLR 3.0.04506.30)",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)",
    "Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0",
    "Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Kazehakase/0.4.5",
    "Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; fr) Presto/2.9.168 Version/11.52",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E; LBBROWSER)",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 LBBROWSER",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E; QQBrowser/7.0.3698.400)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; 360SE)",
    "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)",
    "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.1; WOW64; Trident/5.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C; .NET4.0E)",
    "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1",
    "Mozilla/5.0 (iPad; U; CPU OS 4_2_1 like Mac OS X; zh-cn) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8C148 Safari/6533.18.5",
    "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:2.0b13pre) Gecko/20110307 Firefox/4.0b13pre",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:16.0) Gecko/20100101 Firefox/16.0",
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11",
    "Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"   
]

base_url_en=["https://www.chinadaily.com.cn/","http://en.people.cn/"]
base_url_ch=["http://www.xinhuanet.com","http://www.chinanews.com","http://www.chinadaily.com.cn","http://www.globaltimes.cn","http://www.ecns.cn"]

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
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and depth < 5:
                if href.startswith('/'):
                    href = base_url + href
                elif href.startswith('#'):
                    continue
                if re.match(pattern, href):
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
    crawler_ch()
    crawler_en()
    crawler_en_xinhua()