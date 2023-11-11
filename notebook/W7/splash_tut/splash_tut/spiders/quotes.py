from typing import Iterable

import scrapy
from scrapy import Request
from scrapy_splash import SplashRequest


class CrawlSpider(scrapy.Spider):
    name = "quotes"
    allowed_domains = ["quotes.toscrape.com"]
    start_urls = ["https://quotes.toscrape.com/js/"]

    def start_requests(self) -> Iterable[Request]:
        print("*********** AAAAA")
        for url in self.start_urls:
            print("*********** BBBBB")
            yield SplashRequest(url=url, callback=self.parse, args={'wait': 0.5})

    def parse(self, response, **kwargs):
        print("*********** CCCCCC")
        print(response.body)
        quotes = response.css('div.quote').getall()
        for quote in quotes:
            print("*********** DDDDD")

            yield {
                quote: response.css('span.text::text').get()
            }
