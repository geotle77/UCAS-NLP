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
        next_pages = [a['href'] for a in soup.select("div.swiper-container div.swiper-slide  a")]
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

def extract_chinadaily(link):
    Base_url = "https:"
    url = Base_url + link
    soup = BeautifulSoup(get_html(url), 'html.parser')
    content_divs  = soup.select('div.picshow div#Content p') 
    page_text = ""
    for div in content_divs:
        page_text += div.get_text()
    next_page = soup.select_one('div#currpage a.pagestyle')
    if next_page and 'Next' in next_page.text:
        next_url = next_page['href']
        # If the URL is relative, you might need to add the base URL
        page_text += extract_chinadaily(next_url)
    return page_text


def crawler_chinadaily():
    base_url_en = "https://www.chinadaily.com.cn/china/59b8d010a3108c54ed7dfc23/"
    start_page = 1
    page_num = start_page
    end_page = 4508
    while page_num < end_page:
        url = base_url_en + f"page_{page_num}.html"
        html = get_html(url)
        print(f"parse the index {page_num}")
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.select('div.main_art div.lft_art.lf span.tw3_01_2_p a')
        for link in links:
            link_url = link['href']
            print(f"crawling {link_url}")
            content = extract_chinadaily(link_url)
            save_text(content, 'english.txt')
        page_num += 1
            
def extract_novel(link):
    html = get_html(link)
    soup = BeautifulSoup(html, 'html.parser')
    name = soup.select_one('div#content div.vgvgd').get_text()
    content = ""
    Chpter_links = soup.select('div#content ol.clearfix li a')
    base_url = link+'/'
    chapter_num = 1
    for link in Chpter_links:
        chapter_url = base_url + link['href']
        chapter_html = get_html(chapter_url)
        chapter_soup = BeautifulSoup(chapter_html, 'html.parser')
        chapter_content = chapter_soup.select_one('div.text').get_text()
        print(f"have processed chapter {chapter_num}")
        chapter_num += 1
        content += chapter_content
    return content,name

def crawler_novel():
    novel_url = "https://novel.tingroom.com/count.php"
    start_page = 1
    end_page = 536
    articles = []
    page = start_page
    while page <= end_page:
        url = novel_url + f"?page={page}"
        html = get_html(url)
        soup = BeautifulSoup(html, 'html.parser')
        novel_divs = soup.select('div.all001xp1 div.list')
        links = []
        for div in novel_divs:
            name = div.select_one('div.text h6.yuyu a').get_text()
            link = div.select_one('div.text h6.yuyu a')['href']
            if name not in articles:
                links.append(link)
                articles.append(name)
        base_url = "https://novel.tingroom.com"
        for link in links:
            url = base_url + link
            print(f"crawling {url}")
            content,name = extract_novel(url)
            print(f"saving {name}")
            save_text(content, 'english.txt')
        page += 1
    print(f"total {len(articles)} articles")
        
        
def crawler_en_xinhua():
    base_url = "https://english.news.cn"
    total_size = 0
    max_size = 1024 * 1024 * 100
    q = Queue()
    q.put((base_url, 0))
    pattern = r'^https://english\.news\.cn/\d{8}/.+c\.html$'
    pattern_prefix = r'^https://english\.news\.cn/.+'
    visited = set()
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
                    if href in visited:
                        continue
                    content = extract_xinhua(href)
                    visited.add(href)
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
    with open('./homework1/output/'+filename, 'a', encoding='utf-8') as f:
        f.write(text)



if __name__ == '__main__':
    # crawler_ch()
    # crawler_chinadaily()
    # crawler_en_xinhua()
    crawler_novel()