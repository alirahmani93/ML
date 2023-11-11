import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URI = 'https://digikala.com'
SEARCH_TEXT = 'هوش مصنوعی'

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
driver.get(URI)

search_bar = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, 'base_layout_mobile_sticky_header')))
search_bar.click()

search_bar2 = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.NAME, 'search-input')))

# search_bar = driver.find_element(By.NAME, 'search-input')
search_bar2.send_keys(SEARCH_TEXT)
search_bar2.send_keys(Keys.ENTER)

wrapper_items = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, 'ProductListPagesWrapper')))
if wrapper_items:

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[contains(@class , "product-list_ProductList__item")]//h3')))
    items = wrapper_items.find_elements(By.XPATH, '//div[contains(@class , "product-list_ProductList__item")]//h3')
    for item in items:
        print(item.text)

time.sleep(2)
driver.quit()
