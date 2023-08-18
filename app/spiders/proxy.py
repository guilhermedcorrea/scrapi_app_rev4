from seleniumwire import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import json
import requests

SCRAPEOPS_API_KEY = 'YOUR_API_KEY'
NUM_RETRIES = 2

proxy_options = {
    'proxy': {
        'http': f'http://scrapeops.headless_browser_mode=true:{SCRAPEOPS_API_KEY}@proxy.scrapeops.io:5353',
        'https': f'http://scrapeops.headless_browser_mode=true:{SCRAPEOPS_API_KEY}@proxy.scrapeops.io:5353',
        'no_proxy': 'localhost:127.0.0.1'
    }
}


scraped_quotes = []


url_list = [
    'http://quotes.toscrape.com/page/1/',
    'http://quotes.toscrape.com/page/2/',
]

def get_status_code_first_request(performance_log):
   
    for line in performance_log:
        try:
            json_log = json.loads(line['message'])
            if json_log['message']['method'] == 'Network.responseReceived':
                return json_log['message']['params']['response']['status']
        except:
            pass
    return None

def scrape_with_proxy(driver, url):
    for _ in range(NUM_RETRIES):
        try:
            driver.get(url)
            performance_log = driver.get_log('performance')
            status_code = get_status_code_first_request(performance_log)
            if status_code in [200, 404]:
               
                break
        except requests.exceptions.ConnectionError as e:
            print("error", e)
            driver.close()

    if status_code == 200:
        
        html_response = driver.page_source
        soup = BeautifulSoup(html_response, "html.parser")

       
        quotes_sections = soup.find_all('div', class_="quote")

     
        for quote_block in quotes_sections:
            quote = quote_block.find('span', class_='text').text
            author = quote_block.find('small', class_='author').text
          
            scraped_quotes.append({
                'quote': quote,
                'author': author
            })

if __name__ == "__main__":
  
    option = webdriver.ChromeOptions()
    option.add_argument('--headless') 
    option.add_argument('--no-sandbox')
    option.add_argument('--disable-dev-sh-usage')
    option.add_argument('--blink-settings=imagesEnabled=false')

  
    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}

 
    driver = webdriver.Chrome(ChromeDriverManager().install(), 
                              options=option, 
                              desired_capabilities=caps,
                              seleniumwire_options=proxy_options)

    for url in url_list:
        scrape_with_proxy(driver, url)

    print(scraped_quotes)
