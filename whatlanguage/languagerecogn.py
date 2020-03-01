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
from predict import *

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

#학습 준비
def categoryFromOutput(output):
    top_n, top_i=output.topk(1)#Tensor.topk-텐서의 가장 큰 값의 주소
    category_i=top_i[0].item()
    return all_categories[category_i],category_i
print(categoryFromOutput(output))

def randomChoice(l):
    return l[random.randint(0,len(l)-1)]


#예시학습
def randomTrainingExample():
    category=randomChoice(all_categories)
    line=randomChoice(category_lines[category])
    category_tensor=torch.tensor([all_categories.index(category)],dtype=torch.long)
    line_tensor=lineToTensor(line)
    return category,line,category_tensor,line_tensor

for i in range(10):
    category,line,category_tensor,line_tensor=randomTrainingExample()
    print('category=',category,'/line=',line)

#학습단계
criterion=nn.NLLLoss()

learning_rate=0.005#오차 수정률 정도

def train(category_tensor,line_tensor):
    #은닉층-a다음에 문맥상 b가 올지
    #c가 올지 알 수 없으므로, 보이지 않는 정보도 일부 전달해야 올바른 결과 나옴
    hidden=rnn.initHidden()
    rnn.zero_grad()
    for i in range(line_tensor.size()[0]):
        output,hidden=rnn(line_tensor[i],hidden)

    loss=criterion(output,category_tensor)
    loss.backward()

    #매개변수의 경사도에 학습률을 곱해서 그 매개변수의 값에 더하기
    for p in rnn.parameters():
        p.data.add_(-learning_rate,p.grad.data)

    return output,loss.item()

n_iters=100000
print_every=5000
plot_every=1000

#도식화를 위한 손실 추적
current_loss=0
all_losses=[]

def timeSince(since):
    now=time.time()
    s=now-since
    m=math.floor(s/60)
    s-=m*60
    return '%dm %ds' %(m,s)

start=time.time()

for iter in range(1,n_iters+1):
    category,line,category_tensor,line_tensor=randomTrainingExample()
    output,loss=train(category_tensor,line_tensor)
    current_loss+=loss

    #iter숫자, 손실, 이름, 추측 화면 출력
    if iter%print_every==0:
        guess,guess_i=categoryFromOutput(output)
        correct='✓'if guess==category else '✗ (%s)'%category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start),
                                                loss,line,guess,correct))

    #현재 평균 손실을 전체 손실 리스트에 추가
    if iter%plot_every==0:
        all_losses.append(current_loss/plot_every)
        current_loss=0

# 혼란 행렬에서 정확한 추측을 추적
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000


torch.save(rnn,'char-rnn-classification.pt')