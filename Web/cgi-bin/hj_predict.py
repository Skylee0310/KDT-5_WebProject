# 모듈 로딩
import cgi
import sys
import codecs
import cgitb
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

cgitb.enable()  # Error 확인

# Web 인코딩 설정
sys.stdout = codecs.getwriter(encoding='utf-8')(sys.stdout.detach())

# 웹페이지의 form태그 내의 input 태그 입력값 데이터 가져와서 객체에 저장하고 있는 인스턴스
form = cgi.FieldStorage()

# 클라이언트의 요청 데이터 추출
if 'img_file' in form:
    fileitem = form['img_file']  # form.getvalues('img_file')

    # 서버에 이미지 파일 저장 --------------------------------
    img_file = fileitem.filename

    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')

    save_path = f'./image/{suffix}_{img_file}'
    with open(save_path, 'wb') as f:
        f.write(fileitem.file.read())
    # -------------------------------------------------------

    img_path = f'./image/{suffix}_{img_file}'
else:
    img_path = 'None'

# 모델 클래스 정의 및 로딩
class CNN32(nn.Module):
    def __init__(self):
        super(CNN32, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # 배치 정규화 층 추가
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # 배치 정규화 층 추가

        self.fc1 = nn.Linear(32 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 6)  # 클래스의 수에 맞게 수정

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # 배치 정규화 적용
        x = F.relu(x)
        x = self.pool(x)  # 풀링 적용

        x = self.conv2(x)
        x = self.bn2(x)  # 배치 정규화 적용
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

model = torch.load('model/movie_genre.pt')


file_dir = os.path.dirname(__file__)

def predict_one(filename):
    # 이미지 열기
    # img_path = os.path.join(file_dir, filename)
    img_path = filename
    img = Image.open(img_path)
    preprocessing = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.38588074, 0.3442984, 0.32389283], std=[0.06522466, 0.06430584, 0.073496684])
    ])
    img = preprocessing(img)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(img)
        _, preds = torch.max(output, 1)

    if preds == 0:
        preds = 'Action'
    elif preds == 1:
        preds = 'Animation' 
    elif preds == 2:
        preds = 'Comedy&Drama'
    elif preds == 3:
        preds = 'Horror'
    elif preds == 4:
        preds = 'Romance'
    elif preds == 5:
        preds = 'SF'

    return preds, '.' + img_path

# 이미지 분류 결과
res, img_path = predict_one(img_path)

# 요청에 대한 응답 HTML
# HTML Header
print('Content-Type: text/html; charset=utf-8') # 한글 깨짐 방지 charset=utf-8
print()  # 무조건 한줄 띄어야 함

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
        <form action="../cgi-bin/hj_predict.py" method="post" enctype="multipart/form-data">

            <center>
            <br>
            <br>
            <br>
            <br>
            <br>
                <label><h1>포스터 이미지</h1></label>
                <img src="{img_path}" alt="Movie Poster" width="200" height="300">
                <p style="font-weight: bold; font-size: 24px;">
                    [Predicted Genre]<br>
                
                    {res}
                </p>
            </center>
        </form>
    </center>
</body>
</html>
""")
