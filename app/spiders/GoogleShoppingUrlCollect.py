import scrapy


class GoogleshoppingurlcollectSpider(scrapy.Spider):
    name = "GoogleShoppingUrlCollect"
    allowed_domains = ["shopping.google.com.br"]
    start_urls = ["https://shopping.google.com.br"]

    def parse(self, response):
        pass
