import requests
from bs4 import BeautifulSoup

class WebCrawler:
    def __init__(self, base_url, start_page, end_page, filename):
        self.base_url = base_url
        self.start_page = start_page
        self.end_page = end_page
        self.filename = filename


    def get_html(self, url):
        response = requests.get(url)
        response.encoding = 'utf-8'  # Use 'utf-8' for Chinese websites, 'ISO-8859-1' for English websites
        return response.text

    def parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        # Add your code here to parse the HTML and extract the data you need
        return data

    def clean_data(self, data):
        # Add your code here to clean the data
        return cleaned_data

    def save_data(self, data):
        with open(self.filename, 'a', encoding='utf-8') as f:
            f.write(data + '\n')

    def crawl(self):
        for i in range(self.start_page, self.end_page):
            url = self.base_url.format(i)
            html = self.get_html(url)
            data = self.parse_html(html)
            cleaned_data = self.clean_data(data)
            self.save_data(cleaned_data)