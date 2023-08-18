import scrapy
from undetected_chromedriver import Chrome, ChromeOptions
import random
import time
from sqlalchemy import create_engine, text
from spiders.chrome_options_mixin import ChromeOptionsMixin


class GoogleshoppingpricescollectSpider(ChromeOptionsMixin, scrapy.Spider):
    name = "GoogleShoppingPricesCollect"
    allowed_domains = ["shopping.google.com.br"]
    start_urls = ["https://shopping.google.com.br"]

    def random_delay(self):
        return random.uniform(0.5, 3.0)

    def make_request(self, url, chrome_options):
        with Chrome(options=chrome_options) as driver:
            time.sleep(self.random_delay())
            driver.get(url)
            time.sleep(self.random_delay())

  
    def get_url_google(self, urls, chrome_options):
        for url in urls:
            self.make_request(url['url_google'], chrome_options)

    def parse(self, response):
        chrome_options = self.create_chrome_options()
        self.make_request("https://shopping.google.com.br/", chrome_options)
        
        offset = 0
        limit = 10
        while True:
            urls = self.get_query_database(offset, limit)
            if not urls:
                break

            self.get_url_google(urls, chrome_options)
            offset += limit

         
            with Chrome(options=chrome_options) as driver:
                driver.quit()
