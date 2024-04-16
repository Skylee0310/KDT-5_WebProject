import cgi, sys, codecs
import datetime
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error
import pickle


sys.stdout=codecs.getwriter(encoding='utf-8')(sys.stdout.detach())

# 웹 페이지의 form 태그 내의 input 태그 입력값 가져와서
# 저장하고 있는 인스턴스
form = cgi.FieldStorage()


# 선택한 옵션 값을 가져오기
selected_option = form.getvalue('model')


# 모델 클래스 불러오기
class ScoreRegression(nn.Module):
    def __init__(self,
                 n_vocab,
                 hidden_dim,
                 embedding_dim,
                 n_layers,
                 dropout=0.5,
                 bidirectional=True,
                 model_type='lstm'):
        super(ScoreRegression, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=embedding_dim,
            padding_idx=0)
        
        if model_type == 'rnn':
            self.model = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        elif model_type == 'lstm':
            self.model = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=True,
            )
        if bidirectional:
            self.regression = nn.Linear(hidden_dim*2, 1)
        else:
            self.regression = nn.Linear(hidden_dim,1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        logits = self.regression(last_output)
        
        return logits


ScoreRegression_rnn = torch.load('./cgi-bin/regression_rnn.pth', map_location=torch.device('cpu'))
ScoreRegression_rnn2 = torch.load('./cgi-bin/regression_rnn2.pth', map_location=torch.device('cpu'))
ScoreRegression_lstm = torch.load('./cgi-bin/regression_lstm.pth', map_location=torch.device('cpu'))
ScoreRegression_lstm2 = torch.load('./cgi-bin/regression_lstm2.pth', map_location=torch.device('cpu'))




# 단어사전 불러오기
with open('./vocab1.pkl', 'rb') as f:
    vocab1 = pickle.load(f)


# 입력된 스토리 토큰화 모듈 다운
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK의 불용어 다운로드
nltk.download('stopwords')
nltk.download('punkt')

# 영어의 불용어 목록 가져오기
stop_words = set(stopwords.words('english'))

# 토큰화 함수 생성
def text_pipeline(text):
    tokenList = word_tokenize(text)
    return vocab1(tokenList)

# 패딩함수
def pad_sequences(sentences, maxlen, pad, start='R'):
    result = []
    for sen in sentences:
        sen = sen[:maxlen] if start == 'R' else sen[:maxlen*(-1)]
        padd_sen = sen + [pad] * (maxlen-len(sen)) if start == 'R' else [pad] * (maxlen-len(sen)) + sen
        result.append(padd_sen)
    
    return result


# 인코딩& 디코딩 인덱싱
### 인코딩 : 문자 >>>> 숫자로 변환
token_to_id ={ label : id  for label, id in vocab1.get_stoi().items()}

### 디코딩 : 숫자 >>>> 문자로 변환
id_to_token ={ id : label  for label, id in vocab1.get_stoi().items()}



# data_ids = [[vocab1.get_stoi().get(token,1) for token in text] for text in tokens]




# 예측함수 생성
def predict(model, text, text_pipeline):
    with torch.no_grad():
        # 토큰화 => 정수변환 => 텐서
        text=pad_sequences([text_pipeline(text)],30, 0)
        text = torch.tensor(text)
        logits = model(text)
        return logits.item()



if selected_option == 'rnn':
    model = ScoreRegression_rnn
elif selected_option == 'lstm':
    model = ScoreRegression_lstm
elif selected_option == 'rnn2':
    model = ScoreRegression_lstm
elif selected_option == 'lstm2':
    model = ScoreRegression_lstm


# 텍스트 가져오기
text = form.getvalue('story')



result_score = predict(model,text,text_pipeline)








# 웹에서 뽑아 내기에 이 과정이 필수
print("Content-Type: text/html; charset=utf-8")
print()
# print("<TITLE>스토리점수 회귀분석 결과</TITLE>")
# print("<H1>스토리점수 회귀분석 결과</H1>")
# print(f"<h3>선택한 모델:</h3>")
# print(f'<p>{selected_option}</p><br>')
# print(f"<p>예상 점수 : {round(result_score,2)}</p><br>")

print(f'''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>줄거리로 영화점수 뽑아내기</title>
    <link href ='../css/model.css' rel = "stylesheet">
</head>
<body>
    <center>
    <h2>영화 줄거리 입력</h2>
    <form method="post" action ="/cgi-bin/send_story_data.py" enctype="multipart/form-data">
        <fieldset>
            <p>
                <h3>선택한 모델:</h3>
                <h3>{selected_option}</h3><br>
                <b>예상 점수: {round(result_score,2)}</b><br>
            </p>
            <p>
                <label for="story">내용 : </label>
                <textarea id="story" name="story"></textarea>
            </p>
            <p>
                <label for="model">사용 모델 선택 : </label>
                <select id="model" name="model">
                    <option value="rnn">rnn모델 ver1000</option>
                    <option value="rnn2">rnn모델 ver4000</option>
                    <option value="lstm">lstm모델 ver1000</option>
                    <option value="lstm2">lstm모델 ver4000</option>
                </select>
            </p>
            <input type="submit" value=" " class="submit-button">
        </fieldset>
        
    </form>
    </center>
</body>
</html>''')