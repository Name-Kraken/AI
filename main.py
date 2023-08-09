import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import pytorch_lightning as pl
import transformers

print(transformers.__version__)

df = pd.read_csv(r'C:\Users\royal\Desktop\プログラミング\AI_app\b.df_8instincts.csv')

# 'category_id'列で昇順にソートします
df = df.sort_values('category_id')
# Subtract 1 from all values in the 'category_id' column
df['category_id'] = df['category_id'] - 1

# 分かち書き用の tokenizer
from transformers import BertJapaneseTokenizer
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

text = list(df['title'])[0]
# return_tensors に pt(PyTorch) を選択
wakati_ids = tokenizer.encode(text, return_tensors='pt')
print('各単語に振られている id:', wakati_ids)
print('id のサイズ：', wakati_ids.size())

def bert_tokenizer(text):
    return tokenizer.encode(text, return_tensors='pt')[0]

# Tokenizer の pad トークン ID の取得
pad_token_idx = tokenizer.pad_token_id
print(pad_token_idx)

from torch.nn.utils.rnn import pad_sequence

# テキストデータをトークン化
def translate_index(df, tokenizer):
    text_list = []
    for line in df:
        text_list.append(tokenizer(line))
    text_tensor = pad_sequence(text_list, batch_first=True, padding_value=pad_token_idx)
    return text_tensor

pl.seed_everything(0)
from sklearn.model_selection import train_test_split
df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=0)
print(len(df_train_val))
print(len(df_test))

tensor_train_val = translate_index(df_train_val['title'], bert_tokenizer)
tensor_test = translate_index(df_test['title'], bert_tokenizer)

print(tensor_train_val.shape)
print(tensor_test.shape)

from torch.utils.data import TensorDataset, DataLoader

t_train_val = torch.tensor(df_train_val['category_id'].values, dtype=torch.int64)
t_test = torch.tensor(df_test['category_id'].values, dtype=torch.int64)

dataset_train_val = TensorDataset(tensor_train_val, t_train_val)
dataset_test = TensorDataset(tensor_test, t_test)

n_train = int(len(dataset_train_val)*0.8)
n_val = len(dataset_train_val) - n_train

dataset_train, dataset_val = torch.utils.data.random_split(dataset_train_val, [n_train, n_val])

train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=32)
test_loader = DataLoader(dataset_test, batch_size=32)


from transformers import BertModel
# BERT のモデル構造を確認
bert_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


from torchmetrics.functional import accuracy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BertClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        # BERTの隠れ層の次元数は768, ニュース記事のカテゴリ数が8
        self.fc = nn.Linear(768, 8)

        # Fine tuning の設定
        # 全てを勾配計算 False に設定
        for param in self.parameters():
            param.requires_grad = False

        # Bert の最後の Layer を勾配計算ありに変更
        for param in self.bert.encoder.layer[-1].parameters():
            param.requires_grad = True

        # Linear を勾配計算ありに変更
        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        bert_out = self.bert(x, output_attentions=True)
        # [CLS] に対する分散表現のみ取得
        h = bert_out[0][:,0,:]
        # h = bert_out[1]
        h = self.fc(h)
        return h, bert_out[2]

    def training_step(self, batch, batch_idx):
        x, t = batch
        # forward の出力は out と attentions の tuple
        y = self(x)[0]
        loss = F.cross_entropy(y, t)

        self.log('train_loss', loss, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', accuracy(y, t), logger=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)[0]
        loss = F.cross_entropy(y, t)

        self.log('val_loss', loss, logger=True, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy(y, t), logger=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)[0]
        loss = F.cross_entropy(y, t)

        self.log('test_loss', loss, logger=True, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy(y, t), logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 学習率を分ける
        optimizer = transformers.AdamW([
                                        {'params': self.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
                                        {'params': self.fc.parameters(), 'lr': 3e-5}
        ])
        return optimizer
    

from pytorch_lightning.loggers import CSVLogger
logger = CSVLogger('logs', name='bert_classification')

# 乱数シードの固定
pl.seed_everything(0)
# インスタンス化
net = BertClassifier()
trainer = pl.Trainer(max_epochs=44, gpus=1, logger=logger)
trainer.fit(net, train_loader, val_loader)
result = trainer.test(dataloaders=test_loader)

# ラベルの名前のリストを定義します。
label_names = ["安らぐ本能", "進める本能", "決する本能", "有する本能", "属する本能", "高める本能", "伝える本能", "物語る本能"]

def predict(text):
    # テキストをトークン化
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # GPUが利用可能ならGPUにデータを送る
    if torch.cuda.is_available():
        input_ids = input_ids.to('cuda')
        net.to('cuda')

    # モデルの推論モードをオンにして推論を実行
    net.eval()
    with torch.no_grad():
        outputs = net(input_ids)[0]

    # 各ラベルのスコアを取得し、小数点第2位までに丸める
    scores = [round(float(score), 2) for score in F.softmax(outputs, dim=1)[0]]

    # 最も確率の高いラベルのインデックスを取得
    _, predicted = torch.max(outputs, 1)

    # ラベルのインデックスをPythonのint型に変換
    predicted_label_index = predicted.item()

    # ラベルの名前を取得
    predicted_label_name = label_names[predicted_label_index]

    # 各ラベルのスコアと名前を辞書に格納
    label_scores = {label_names[i]: score for i, score in enumerate(scores)}

    return predicted_label_name, label_scores
import streamlit as st

# StreamlitアプリのUI部分
st.title('どの本能活性化されている？')
# ライブラリ追加
from PIL import Image

img = Image.open(r'C:\Users\royal\Desktop\プログラミング\AI_app\logo.jpg')

# use_column_width 実際のレイアウトの横幅に合わせる
st.image(img, caption='', use_column_width=True)

st.text('参考文献')
st.text('著：鈴木 祐【ヒトが持つ8つの本能に刺さる進化論マーケティング】')
st.text('テキストによって何の本能が活性化されているのか調べることが出来ます')

text = st.text_area("テキストを入力してください:", value='', max_chars=None, key=None)

if st.button('予測'):
    predicted_label, label_scores = predict(text)
    st.write(f"最も活性化されている本能: {predicted_label}")
    st.write("各ラベルのスコア:")
    for label, score in label_scores.items():
        st.write(f"{label}: {score}")

    # ラベルのスコアをPandas DataFrameに変換
    scores_df = pd.DataFrame(list(label_scores.items()), columns=['Label', 'Score'])
    scores_df = scores_df.set_index('Label')

    # 棒グラフで表示
    st.bar_chart(scores_df)