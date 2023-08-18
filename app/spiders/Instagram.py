import scrapy


class InstagramSpider(scrapy.Spider):
    name = "Instagram"
    allowed_domains = ["instagram.com"]
    start_urls = ["https://instagram.com"]

    def parse(self, response):
        pass
