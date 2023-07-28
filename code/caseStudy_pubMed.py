# -*- coding: utf-8 -*-
# @Author  : Zhanjianming
# @Time    : 2023/7/28 19:32
# @File    : caseStudy_pubMed.py
# @Software: PyCharm

from selenium import webdriver
from selenium.webdriver import ActionChains
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Get drug combination organization name
novel_pred = pd.read_csv('Record/case_study_predNovel/top_pred_select_100.csv')
novel_pred = novel_pred.drop(['Unnamed: 0'], axis=1)

# Manually query pubmed
# Add new column
novel_pred['evidence'] = ''
# Start query
for i in range(len(novel_pred)):
    try:
        drug1 = novel_pred['drugA'][i]
        drug2 = novel_pred['drugB'][i]
        # Set up the Chrome browser
        driver = webdriver.Chrome(executable_path=r'chromedriver.exe')
        driver.get("https://pubmed.ncbi.nlm.nih.gov/?term=%28"+drug1+"%29+AND+%28"+drug2+"%29")
        # Give the browser time to open the page
        time.sleep(5)
        result = driver.find_element_by_class_name("top-wrapper").find_element_by_class_name("results-amount-container").find_element_by_class_name("results-amount")
        content = result.find_element(by='tag name', value='h3').text
        novel_pred.loc[i,'evidence'] = content
        print("id:" + str(i) + " evidence:" + content)
    except Exception as e:
        print("id:" + str(i) + " evidence:1")
        novel_pred.loc[i,'evidence'] = '1'
    finally:
        # close the window
        novel_pred.to_csv('Record/case_study_predNovel/top_pred_select_.csv')
        driver.quit()

# save content
novel_pred.to_csv('Record/case_study_predNovel/top_pred_select_.csv')
print("success!!!")