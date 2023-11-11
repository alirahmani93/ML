from typing import Iterable

import scrapy
from scrapy import Request


class DivarSpider(scrapy.Spider):
    name = 'divar'
    allowed_domains = ['api.divar.ir']
    city = 'tehran'
    start_urls = ['https://api.divar.ir/v8/web-search/tehran/buy-residential']
