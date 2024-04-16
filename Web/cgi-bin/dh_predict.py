# 모듈 로딩
import cgi, cgitb, sys, codecs, datetime
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

cgitb.enable()

# WEB 인코딩 설정
sys.stdout = codecs. getwriter(encoding = 'utf-8')(sys.stdout.detach())

# resnet18 모델 인스턴스 생성
MY_MODEL = resnet18()

# 전결합층 변경
# in feature : FeatureMap에서 받은 피처 수, out_featrues : 출력/분류 클래스 수
MY_MODEL.fc = nn.Sequential(nn.Linear(512, 128),
              nn.BatchNorm1d(num_features = 128),
              nn.ReLU(),
              nn.Dropout(),
              nn.Linear(128, 32),
              nn.BatchNorm1d(num_features = 32),
              nn.ReLU(),
              nn.Dropout(),
              nn.Linear(32, 1))

# 모델 불러오기
my_model = torch.load('./model/ResNet18_Member_Score3.pt')

# 예측 함수
def predict(image_data):
    
    with torch.no_grad():
        output = my_model(image_data)
        
    return output


# client 요청 데이터 즉, form 데이터 저장 인스턴스
form = cgi.FieldStorage()

preprocessing = transforms.Compose([transforms.Resize(size = (64, 64)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))])

# 클라이언트의 요청 데이터 추출
if 'img_file' in form:
    
    fileitem = form['img_file']  # form.getvalue(key = 'img_file')
    if fileitem is not None:
    # 서버에 이미지 파일 저장
    # ext = '.' + fileitem.filename.rsplit('.')[-1]
        img_file = fileitem.filename
    
        suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

        save_path = f'./image/{suffix}_{img_file}'
        with open(save_path, mode = 'wb') as f:
            f.write(fileitem.file.read())

        img_path = f'../image/{suffix}_{img_file}'
    else:
        img_path = None
else:
    img_path = None
   

if img_path is not None:
    data = Image.open(save_path)
    data = preprocessing(data)
    data = data.unsqueeze(0)
    prediction = predict(data).item()
    result = f'모델이 예측한 회원 점수는 {round(prediction, 1)}입니다.'
else:
    result = '결과 없음'

# Web 브라우저 화면 출력 코드
# file_path = 'ex_img_input2.html'

# HTML Body
print(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="../CSS/model.css" rel="stylesheet">

<style>
      h1 {{
          padding:15px
      }}
</style>
      
    <title>Genres Classify</title>
</head>
<body>
    <center>
        <form action="../cgi-bin/dh_predict.py" method="post" enctype="multipart/form-data">

            <center>
            <br>
            <br>
            <br>
            <br>
            <br>
                <label><h1>포스터 이미지</h1></label>
                <img src="{img_path}" alt="Movie Poster" width="200" height="300">
                <p style="font-weight: bold; font-size: 24px;">
                    [Predicted Score]<br>
                
                    {result}
                </p>
            </center>
        </form>
    </center>
</body>
</html>
""")