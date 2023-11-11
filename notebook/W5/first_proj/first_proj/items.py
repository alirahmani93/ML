# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html
from dataclasses import dataclass

import scrapy


class FirstProjItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass


def serializer_status(value):
    if value.contains('باز است'):
        return True
    return False


def serializer_restaurants_name(value):
    pass


def serializer_restaurants_address(value):
    pass


class FidioRestaruantItem(scrapy.Item):
    status = scrapy.Field(serializer=serializer_status)
    name = scrapy.Field(serializer=serializer_restaurants_name)
    address = scrapy.Field(serializer=serializer_restaurants_address)


@dataclass
class RestaurantDataClass:
    name: str
    address: str
    status: str
