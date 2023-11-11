from pathlib import Path

import scrapy


class JobVisionSpider(scrapy.Spider):
    name = "job_vision"
    allowed_domains = 'https://jobvision.ir'
    start_urls = [
        "https://jobvision.ir/jobs/category/in-tehran?keyword=data%20analyst&page=1&sort=1",
    ]

    def start_requests(self):
        urls = [
            "https://jobvision.ir/jobs/category/in-tehran?keyword=data%20analyst&page=1&sort=1",
            # "https://quotes.toscrape.com/page/2/",
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):
        page = response.url.split("/")[-2]
        filename = f"quotes-{page}.html"
        Path(filename).write_bytes(response.body)
        self.log(f"Saved file {filename}")
