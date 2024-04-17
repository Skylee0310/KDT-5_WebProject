# KDT-5_WebProject
경북대학교 KDT(Korea Digital Training) Web 프로젝트

## 영화 장르 및 평점 예측 모델

  
#### DATA
- TMDB
https://www.themoviedb.org/
  
#### 역할분담 및 기능설계서

![image](https://github.com/KDT5-1TEAM/KDT-5_NLPProject/assets/155441547/64e09c2a-66d9-4994-b100-3985fbf78d0a)






<details>
  <summary>
    박희진( 포스터로 장르 예측 모델 )
  </summary>
  
  ## (1) 데이터 생성 , 확인 및 전처리

- tmdb 웹사이트에서 크롤링
    - 웹사이트가 동적 웹사이트여서 무한 스크롤 하면서 웹 크롤링 해야 했음. 끝 페이지에 도달했을 때, 계속 페이지가 닫겨서 결국 끝까지 스크롤하고 각 영화 정보 링크를 수집해서 파일 생성
    - movie_path.csv
- 포스터 이미지 링크, 제목, 장르, 러닝타임, 회원점수, 줄거리, 제작비,  수익, 감독 데이터 크롤링
    - 셀레늄 이용
    - try구문 이용하여 값이 없는 경우 None값이 입력되게 설정
    - 장르는 여러 개인 경우가 있어서 모든 장르 요소 크롤링 후 리스트 안에 담은 후 저장
    - 데이터프레임으로 저장 후 csv 파일로 변환
        - movie_data.csv
        - 약 10000개의 영화 데이터
- pd.read_csv로 movie_data.csv 불러오기
- 필요한 데이터는 포스터 이미지 링크와 장르기 때문에 포스터 이미지 링크와 장르가 None값인 행 삭제
- 데이터프레임에서 이미지 URL과 장르를 추출
    - 장르를 set안에 담아 장르 종류 확인
- *Animation > Horror > SF > Romance > Action > Comedy&Drama* 를 우선순위로 두고 여러 장르를 가진 영화를 하나의 장르로 함축
    - 포스터에서 특징이 강할 것 같은 장르 순으로 임의로 지정
- 포스터 URL과 장르 각 시리즈 concat을 통해 결합
- 유니크한 장르를 확인하고 각 라벨의 분포를 시각화

![image]운 점

- 장르는 보통 국한되지 않음. 다중 레이블 분류 모델을 만들었다면 더욱 성능이 좋았을 것.

## (10) 피드백 : 모두에게

- 주제에 대한 타당성을 생각해라. 주제에 데이터가 적합한지 왜 주제를 그것으로 잡았는지 어떤 인사이트를 도출하기 위함인지에 대해 고민해봐라. 논문도 찾아보고 기술에 대한 고민도 해볼 것. 새로운 기술에 대해 지금 당장 이해가 되지 않아도 사용하면서 공부해라. 찾아보는 것도 능력이고, 새로운 기술을 이용하는 것도 능력, 새로운 기술을 빨리 익히는 것도 능력이다.
</details>
  
<details>
  <summary>
    김동현( 주제 )
  </summary>
  
</details>
  
<details>
  <summary>
    이화은( 주제 )
  </summary>
</details>


<details>
  <summary>
    양현우( 주제 )
  </summary>
</details>
