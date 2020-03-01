
from languagerecogn import*
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