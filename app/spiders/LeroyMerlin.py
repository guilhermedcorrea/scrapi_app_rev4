import scrapy
from undetected_chromedriver import Chrome, ChromeOptions
import random
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from typing import Any

from spiders.chrome_options_mixin import ChromeOptionsMixin




class LeroymerlinSpider(scrapy.Spider, ChromeOptionsMixin):
    name = "LeroyMerlin"
    allowed_domains = ["leroymerlin.com.br"]
    start_urls = ["https://leroymerlin.com.br"]

    base_url = "https://www.leroymerlin.com.br/porcelanatos?term=porcelanato&searchTerm=porcelanato&searchType=Shortcut&page="

    def click_last_page_button(self, driver: Any) -> (int | None):
        try:
            button = driver.find_element(By.XPATH, "/html/body/div[7]/div[4]/div[1]/div[2]/div[4]/nav/button[2]/i")
            button.click()
            WebDriverWait(driver, 10).until(EC.staleness_of(button))

            last_page_url = driver.current_url
            last_page_number = int(last_page_url.split("page=")[-1])

            return last_page_number

        except Exception as e:
            print("Erro ao clicar no botão da última página:", e)
            return None

    def click_and_get_urls(self, driver: Any) -> (list[Any] | list):
        try:
            urls_products = driver.find_elements(By.XPATH, "/html/body/div/div/div/div/div/div/div/div/div/div/a")
            return [urls.get_attribute("href") for urls in urls_products]
        except Exception as e:
            print("Erro ao obter URLs:", e)
            return []

    def extract_product_details(self, driver: Any) -> dict:
        driver.implicitly_wait(30)
        product_dict = {}

        try:
            nome = driver.find_elements(By.XPATH, "/html/body/div[10]/div/div[1]/div[1]/div/div[1]/h1")[0].text
            product_dict["nome"] = nome
        except:
            pass

        try:
            precos = driver.find_elements(By.XPATH, "/html/body/div[10]/div/div[1]/div[2]/div[2]/div/div[1]/div/div[2]/div[2]/div/span[1]")[0].text
            product_dict["precos"] = float(precos.replace("R$", "").replace(",", "").replace(",", ".").strip())
        except:
            pass

        try:
            preco_detalhes = driver.find_elements(By.XPATH, "/html/body/div[10]/div/div[1]/div[2]/div[2]/div/div[1]/div/div[3]/div/strong")[0].text
            product_dict["detalhespreco"] = preco_detalhes
        except:
            pass
        
       
        return product_dict

    def click_last_page_button(driver: Any) -> (int | None):
        
        try:
            button = driver.find_element(By.XPATH, "/html/body/div[7]/div[4]/div[1]/div[2]/div[4]/nav/button[2]/i")
            button.click()
            WebDriverWait(driver, 10).until(EC.staleness_of(button))
               
            last_page_url = driver.current_url
            last_page_number = int(last_page_url.split("page=")[-1])

            return last_page_number

        except Exception as e:
            print("Erro ao clicar no botão da última página:", e)
            return None

    last_page_number = click_last_page_button(driver)

    all_urls = []
'''
    if last_page_number is not None:
        
        
        for page_number in range(1, last_page_number + 1):
            page_url = base_url + str(page_number)
            make_request(driver, page_url)
            page_urls = click_and_get_urls(driver)
            all_urls.extend(page_urls)

        print("URLs coletadas:")
        for url in all_urls:
            print(url)

        products = []

        for url in all_urls:
            make_request(driver, url)
            product = extract_product_details(driver)
            
            insert_or_update_products(nome=product['nome'],detalhespreco=product['detalhespreco']
                                      ,descricao=product['descricao'],precos=product['precos'])
            
            
            products.append(product)

        print("Detalhes dos produtos:")
        for product in products:
            ...
            

        finally:
            driver.quit()
'''