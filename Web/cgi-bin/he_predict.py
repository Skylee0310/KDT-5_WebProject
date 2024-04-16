# 모듈 로딩
import cgi, sys, codecs, cgitb
import torch
import pandas as pd
from torch import nn as nn

sys.stdout=codecs.getwriter(encoding='utf-8')(sys.stdout.detach())

# 웹페이지의 form 태그 내의 input 태그 입력값 가져와서 
# 저장하고 있는 인스턴스
form = cgi.FieldStorage()



class LSTM_with_dropout(nn.Module) :
    def __init__(self, n_vocab, hidden_dim, embedding_dim, n_layers, dropout, bidirectional=False) :
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings = n_vocab,
                                      embedding_dim = embedding_dim,
                                      padding_idx = 0)
        
        self.model = nn.LSTM(input_size = embedding_dim,
                             hidden_size = hidden_dim,
                             num_layers = n_layers,
                             bidirectional=bidirectional,
                             dropout = dropout,
                             batch_first = True)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs) :
        embeddings = self.embedding(inputs)
        output, _ = self.model(embeddings)
        last_output = output[:, -1, :]
        logits = self.classifier(last_output)
        return logits

mdl = torch.load('./model/lstm.pth')

df = pd.read_csv('./encoded.csv')
vocabDF = pd.read_csv('./vocab.csv')

# 토큰화 함수 생성
def tokentext(text, df, vocaDF) :
    import spacy
    import re
    from spacy.lang.en.stop_words import STOP_WORDS
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    textToken =[token.text for token in doc]

    voca_dict = {word: idx for idx, word in enumerate(vocaDF['0'].to_list())}
    encoded = [voca_dict[token] for token in textToken if token in voca_dict]

    # 인코딩 결과 출력
    # print("토큰화된 단어들:", textToken)
    # print("인코딩 결과:", encoded)

    padded_id=[]
    max_length = df.shape[1]

    if len(encoded) < max_length :
        padded_id.append(encoded + [0]*(max_length-len(encoded)))
    else :
        padded_id.append(encoded)
    return padded_id[0]

def predict(model, text):
    with torch.no_grad():
        text = torch.tensor((text), dtype=torch.int64)
        text = text.unsqueeze(0)
        predicted_label = model(text)
        return predicted_label.argmax(1).item()

def print_browser(result=""):
    # HTML 파일 읽기 -> body 문자열
    filename = 'he_result.html'
    with open(file=filename, mode = 'r', encoding='utf-8') as f :

        # HTML Header
        print('content-Type : text/html; charset=utf-8')
        print() # header와 body 구분 - 없으면 출력 불가능

        # HTML Body
        html_body = f.read().format(result)
        print()
        print(html_body)

# 텍스트 가져오기
text = form.getvalue('plot')

text = tokentext(text, df, vocabDF)

result = predict(mdl,text)

if result == 0 : result = 'OTHER GENRE'
elif result == 1 : result = 'ROMANCE GENRE'


# # #웹에서 뽑아 내기에 이 과정이 필수
# print("Content-Type: text/html; charset=utf-8")
# print('<link href ="./css/my_style.css" rel = "stylesheet">')

# print()
# print("<TITLE>줄거리로 영화 장르 분류하기</TITLE>")
# print(f"<p class = 're'> 장르 : {result}</p><br>")
print_browser(result)