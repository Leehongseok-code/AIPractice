from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math

def findFiles(path):return glob.glob(path)

print(findFiles('data/names/*.txt'))

all_letters=string.ascii_letters+".,;'"
n_letters=len(all_letters)

#유니코드 문자열을 ASCII코드로 변환
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c)!='Mn'
        and c in all_letters
    )
print(unicodeToAscii('choiminsik'))

#각 언어의 이름 목록인 category_lines 사전 생성
category_lines={}
all_categories=[]

#파일을 읽고 줄 단위로 분리
def readLines(filename):
    lines=open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category= os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines=readLines(filename)
    category_lines[category]=lines
n_categories=len(all_categories)
#test print
print(category_lines['Italian'][:5])

'''
One-Hot벡터는 언어를 다룰 때 자주 이용,
단어, 글자 등을 벡터로 표현 시 단어, 글자 사이의 상관 관계를 미리 알 수 없을 경우,
One-Hot 벡터로 표현하여 서로 직교한다고 가정하고 학습을 시작.
'''

#all_letters 로 문자의 주소 찾기, ex)"a"=0
def letterToIndex(letter):
    return all_letters.find(letter)

#검증을 위해서 한개의 문자를 <1 x n_letters> Tensor로 변환
def letterToTensor(letter):
    tensor=torch.zeros(1,n_letters)
    tensor[0][letterToIndex(letter)]=1
    return tensor

#한 줄(이름)을 <line_length x 1 x n_letters>,
#또는 One-Hot 문자 벡터의 Array로 변경 ex)"apple"-[1,16,16,12,5]
def lineToTensor(line):
    tensor=torch.zeros(len(line),1,n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)]=1
    return tensor

print(letterToTensor('J'))
print(lineToTensor('Jones').size())

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        #입력-input_size+hidden_size의 차원을 가지는 벡터, 출력-hidden_size의 차원을 가지는 벡터
        self.i2h=nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o=nn.Linear(input_size+hidden_size,output_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        combined=torch.cat((input,hidden),1)
        hidden=self.i2h(combined)
        output=self.i2o(combined)
        output=self.softmax(output)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1,self.hidden_size)

n_hidden=128
rnn=RNN(n_letters,n_hidden,n_categories)

input=letterToTensor('A')
hidden=torch.zeros(1,n_hidden)
output,next_hidden=rnn(input,hidden)

#효율성을 위해 매 단계 새로운 텐서를 만드는 대신 lineToTensor로잘라서 사용
input=lineToTensor('Albert')
hidden=torch.zeros(1,n_hidden)

output,next_hidden=rnn(input[0],hidden)
print(output)
