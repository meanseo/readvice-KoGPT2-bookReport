from konlpy.tag import Okt
from nltk.tokenize import word_tokenize
import nltk
import re
import pandas as pd
from nltk import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from icecream import ic
from collections import Counter

class Solution():
    def __init__(self):
        self.okt = Okt()
        self.texts = None

    def hook(self):
        def print_menu():
            print('0. Exit')
            print('1. nltk 다운로드')
            print('2. 전처리')
            print('3. 워드클라우드')
            return input('메뉴 선택 \n')

        while 1:
            menu = print_menu()
            if menu == '0':
                break
            elif menu == '1':
                Solution.download()
            elif menu == '2':
                _ = self.preprocessing()
                ic(_)
            elif menu == '3':
                self.draw_wordcloud()

    @staticmethod
    def download():
        nltk.download('punkt')

    def preprocessing(self):
        texts = self.texts
        with open('./data/book_report_data.txt', 'r', encoding='utf-8') as f:
            texts = f.read()
        texts = texts.replace('\n', ' ')
        tokenizer = re.compile(r'[^ㄱ-힣]+')
        return tokenizer.sub(' ', texts)

    def draw_wordcloud(self):
        nouns = self.okt.nouns(self.texts) # 명사만 추출
        words = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외
        c = Counter(words)
        wc = WordCloud(font_path='malgun', width=400, height=400, scale=2.0, max_font_size=250)
        gen = wc.generate_from_frequencies(c)
        plt.figure()
        plt.imshow(gen)

if __name__ == '__main__':
    Solution().hook()