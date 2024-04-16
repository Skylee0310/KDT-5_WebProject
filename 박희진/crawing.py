import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import re
import time
import csv
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver import ActionChains

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--user-data-dir=/path/to/your/chrome/profile")
chrome_options.add_argument("--disable-extensions")  # 확장 프로그램 비활성화

driver = webdriver.Chrome(options=chrome_options)

fieldnames = ['포스터', '제목', '장르', '러닝타임', '회원점수', '줄거리', '제작비', '수익', '감독']


# CSV 파일 열기
# with open('movie_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     # 헤더 추가
#     writer.writeheader()

path = pd.read_csv('movie_path.csv', sep='\t')
paths = path['-href']

# 각자 할 것  
hj_movie_paths = paths[2128:2500]
hw_movie_paths= paths[2500:5000]
ha_movie_paths = paths[5000:7500]
dh_movie_paths = paths[7500:10000]

# ex ) for path in paths => for path in hw_movie_paths:
# 데이터 수집
for path in hj_movie_paths:
    driver.get(path)
    time.sleep(2)
    # 각 정보를 담을 리스트
    poster_list = []
    title_list = []
    genre_list = []
    duration_list = []
    score_list = []
    overview_list = []
    budget_list = []
    revenue_list = []
    director_list = []

    # 포스터 
    try:
        poster = driver.find_element(By.XPATH, '//*[@id="original_header"]/div[1]/div/div[1]/div/img').get_attribute('src')
        print(poster)
        poster_list.append(poster)
    except:
        poster = None
        print(poster)
        poster_list.append(poster)

    # 제목
    try:
        title = driver.find_element(By.XPATH, '//*[@id="original_header"]/div[2]/section/div[1]/h2/a').text
        print(title)
        title_list.append(title)
    except:
        title = None
        print(title)
        title_list.append(title)

    # 장르
    try:
        glist = []
        genres_elements = driver.find_elements(By.XPATH, './/a[starts-with(@href, "/genre/")]')
        for genre_element in genres_elements:
            glist.append(genre_element.text)
        print(glist)
        genre_list.append(', '.join(glist))
    except:
        genre_list.append(None)

    # 러닝타임
    try:
        duration = driver.find_element(By.CLASS_NAME, 'runtime').text
        print(duration)
        duration_list.append(duration)
    except:
        duration = None
        print(duration)
        duration_list.append(duration)

    # 회원 점수
    try:
        score_element = driver.find_element(By.XPATH, '//*[@id="consensus_pill"]/div/div[1]/div/div/div/span').get_attribute('class')
        score_value = int(re.search(r'\d+', score_element).group()) # 숫자만 추출하여 int형으로 저장
        print(score_value)
        score_list.append(score_value)
    except:
        score_value = None
        print(score_value)
        score_list.append(score_value)

    # 개요
    try:
        overview = driver.find_element(By.XPATH, '//*[@id="original_header"]/div[2]/section/div[3]/div/p').text
        print(overview)
        overview_list.append(overview)
    except:
        overview = None
        print(overview)
        overview_list.append(overview)

    # 제작비
    try:
        budget_element = driver.find_element(By.XPATH, '//*[@id="media_v4"]/div/div/div[2]/div/section/div[1]/div/section[1]/p[4]').text
        budget_text = re.sub(r'[^\d.]', '', budget_element)
        budget_value = float(budget_text.rsplit('.', 1)[0] + '.' + budget_text.rsplit('.', 1)[-1])
        print(budget_value)
        budget_list.append(budget_value)
    except:
        budget_value = None
        print(budget_value)
        budget_list.append(budget_value)

    # 수익
    try:
        revenue_element = driver.find_element(By.XPATH, '//*[@id="media_v4"]/div/div/div[2]/div/section/div[1]/div/section[1]/p[5]').text
        revenue_text = re.sub(r'[^\d.]', '', revenue_element)
        revenue_value = float(revenue_text.rsplit('.', 1)[0] + '.' + revenue_text.rsplit('.', 1)[-1])
        print(revenue_value)
        revenue_list.append(revenue_value)
    except:
        revenue_value = None
        print(revenue_value)
        revenue_list.append(revenue_value)

    
    # 감독
    try:
        director = driver.find_element(By.XPATH, '//*[@id="original_header"]/div[2]/section/div[3]/ol/li[1]/p[1]').text
        print(director)
        director_list.append(director)
    except:
        director = None
        print(director)
        director_list.append(director)
    
    # 데이터 프레임 생성
    df = pd.DataFrame({
        '포스터': poster_list,
        '제목': title_list,
        '장르': genre_list,
        '러닝타임': duration_list,
        '회원점수': score_list,
        '개요': overview_list,
        '제작비': budget_list,
        '수익': revenue_list,
        '감독': director_list
    })

    # 데이터 프레임 저장
    # df.to_csv('movie_data.csv', mode='a', index=False, header=False)


