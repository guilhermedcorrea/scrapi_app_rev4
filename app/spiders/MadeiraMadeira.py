import scrapy


class MadeiramadeiraSpider(scrapy.Spider):
    name = "MadeiraMadeira"
    allowed_domains = ["madeiramadeira.com.br"]
    start_urls = ["https://madeiramadeira.com.br"]

    def parse(self, response):
        pass
