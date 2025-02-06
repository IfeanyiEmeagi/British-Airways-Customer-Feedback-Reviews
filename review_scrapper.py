# The program scraps the reviews and scores assigned to British Airways on Skytrax

import scrapy
from scrapy.crawler import CrawlerProcess


class ReviewSpider(scrapy.Spider):
    name = "BA Reviews"
    page_size = 100
    start_page = 1
    base_url = "https://www.airlinequality.com/airline-reviews/british-airways"
    start_urls = [
        f"{base_url}/page/{start_page}/?sortby=post_date%3ADesc&pagesize={page_size}"
    ]

    def parse(self, response):
        for review in response.css("article[itemprop='review']"):
            review_content = review.css(
                "div.text_content[itemprop='reviewBody'] *::text"
            ).getall()
            yield {
                "title": review.css("h2.text_header::text").get(),
                "review": " ".join(review_content).strip(),
                "score": review.css('span[itemprop="ratingValue"]::text').get(),
            }

        # Corrected CSS selector to get the current page number
        current_page_number = response.css("article ul li a.active::text").get()
        if current_page_number:
            next_page_number = int(current_page_number) + 1
            url = (
                self.base_url
                + f"/page/{next_page_number}/?sortby=post_date%3ADesc&pagesize={self.page_size}"
            )
            yield scrapy.Request(url, callback=self.parse)
        else:
            print("Reached the last page.")


# Run the script as a standalone Python script.
if __name__ == "__main__":
    process = CrawlerProcess(
        settings={"FEEDS": {"ba_reviews.json": {"format": "json", "encoding": "utf-8"}}}
    )
    process.crawl(ReviewSpider)
    process.start()
