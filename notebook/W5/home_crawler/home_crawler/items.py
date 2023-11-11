# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class HomeCrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    title = scrapy.Field()
    image_count = scrapy.Field()
    top_description_text = scrapy.Field()
    middle_description_text = scrapy.Field()
    bottom_description_text = scrapy.Field()
    payload_title = scrapy.Field()
    district_persian = scrapy.Field()
    city_persian = scrapy.Field()
    category_slug_persian = scrapy.Field()
    token = scrapy.Field()
    price = scrapy.Field()
