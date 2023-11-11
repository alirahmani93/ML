from typing import Iterable

import scrapy
from scrapy import Request


class DivarSpider(scrapy.Spider):
    name = 'divar'
    allowed_domains = ['api.divar.ir']
    city = 'tehran'
    start_urls = ['https://api.divar.ir/v8/web-search/tehran/buy-residential']

    def start_requests(self) -> Iterable[Request]:
        # tokens =
        # self.start_urls = [uri.format(self.city) for uri in self.start_urls]
        print('++++++++++++++++')
        print(self.start_urls)
        return super().start_requests()

    def parse(self, response, **kwargs):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        posts = response
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        for post in posts:
            yield Request('https://divar.ir' + post, callback=self.parse_post, dont_filter=True)

    def parse_post(self, response):
        print('################################################')
        # post_header_title = response.xpath('//*[@class="post-header__title"]/text()').extract_first()
        #
        # fields_available_title = response.xpath('//*[@class="post-fields-item__title"]/text()').extract()
        # fields_available_value = response.xpath('//*[@class="post-fields-item__value"]/text()').extract()
        # title_value_dict = dict(zip(fields_available_title, fields_available_value))
        # yield {'post_title': post_header_title}
        # for key, value in list(title_value_dict.items()):
        #     yield {key: value}
