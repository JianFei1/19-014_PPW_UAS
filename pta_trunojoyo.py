import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):

        arrayData = ['https://pta.trunojoyo.ac.id/c_search/byprod/7']
        for i in range(2, 12):
            inArray = 'https://pta.trunojoyo.ac.id/c_search/byprod/7/' + str(i)
            arrayData.append(inArray)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1,6):
            yield {
                'link': response.css('#content_journal > ul > li:nth-child(' + str(i) + ') > div:nth-child(3) > a::attr(href)').extract()
            }
