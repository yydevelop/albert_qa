# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().system('apt-get -q -y install sudo file mecab libmecab-dev mecab-ipadic-utf8 git curl python-mecab > /dev/null')
get_ipython().system('pip install mecab')
get_ipython().system('pip install mecab-python3')
get_ipython().system('git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git > /dev/null ')
get_ipython().system('echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n > /dev/null 2>&1')
get_ipython().system('ln -s /etc/mecabrc /usr/local/etc/mecabrc')


# %%
import csv
import json
import MeCab
import pandas as pd


# %%
def conv_neologd(input_file, output_file):
	# ファイルをオープン
	with open(input_file, "r", encoding='utf-8') as f:
		train = json.load(f)

	# Mecabの定義
	mtagger = MeCab.Tagger("-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")

	# JSONの要素ごとに分解
	# 要素ごとに取得、分かち書き、更新を行う
	for data in train["data"]:
		paragraphs = data["paragraphs"]
		for paragraph in paragraphs:
			# 本文(context)
			context = paragraph["context"]
			context = mtagger.parse(context).replace(' \n', '')
			paragraph["context"] = context
			for qas in paragraph["qas"]:
				# 質問(question)
				question = qas["question"]
				question = mtagger.parse(question).replace(' \n', '')
				qas["question"] = question

				# 回答(text)
				text = qas["answers"][0]["text"]
				text = mtagger.parse(text).replace(' \n', '')
				qas["answers"][0]["text"] = text

				# 回答位置(answar_start)
				# 回答(text)が本文(context)内のどこにあるのかを探索して更新
				qas["answers"][0]["answer_start"] = context.find(text)

	# ファイル出力
	with open(output_file, 'w', encoding='utf-8') as f:
	  json.dump(train, f, ensure_ascii=False)


# %%
from google.colab import drive
drive.mount('/content/gdrive')


# %%
input_file1 = "/content/gdrive/MyDrive/Colab Notebooks/drive_data/DDQA-1.0_RC-QA_dev.json"
output_file1 = "/content/gdrive/MyDrive/Colab Notebooks/drive_data/DDQA-1.0_RC-QA_dev_neologd.json"
input_file2 = "/content/gdrive/MyDrive/Colab Notebooks/drive_data/DDQA-1.0_RC-QA_train.json"
output_file2 = "/content/gdrive/MyDrive/Colab Notebooks/drive_data/DDQA-1.0_RC-QA_train_neologd.json"


# %%
conv_neologd(input_file1, output_file1)
conv_neologd(input_file2, output_file2)


