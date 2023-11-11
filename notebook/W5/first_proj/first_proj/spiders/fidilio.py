import scrapy

from notebook.W5.first_proj.first_proj.items import FidioRestaruantItem, RestaurantDataClass


class FidilioSpider(scrapy.Spider):
    name = "Fidilio"
    allowed_domains = ["fidilio.com"]
    start_urls = ["https://fidilio.com/restaurants/in/tehran/تهران/"]

    def parse(self, response, **kwargs):
        item_per_page = response.css('justify-self-center.justify-center.items-center.mt-6.min-w-fit.h-[350px]')
        for item in item_per_page:
            href = item.css('a::attr("href")').get()
            if href:
                yield response.follow(response.urljoin(href), callback=self.pars_detail_restaurant)
        next_page = response.css(
            'bg-fidilio-red.w-full.h-[45px].text-white.py-2.px-4.rounded-full').xpath('..').attrib['href']
        if next_page:
            yield scrapy.Request(response.urljoin(next_page))

    def pars_detail_restaurant(self, response):
        status = response.css('p.text-white.text-sm').get()
        restaurants_name = response.css('p.text-lg.venue-name.line-clamp-1::text').get()
        restaurants_address = response.css('p.text-fidilio-dark-gray.mx-2.line-clamp-2::text').get()
        if status is not None:
            yield RestaurantDataClass(
                status=status, restaurants_name=restaurants_name, restaurants_address=restaurants_address, )
            # item = FidioRestaruantItem()
            # item['status'] = status
            # item['restaurants_name'] = restaurants_name
            # item['restaurants_address'] = restaurants_address

            # yield item
            # yield {
            #     "status": status,
            #     "restaurants_name": restaurants_name,
            #     "restaurants_address": restaurants_address}
