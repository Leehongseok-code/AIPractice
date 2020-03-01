from languagerecogn import *


rnn=torch.load('char-rnn-classification.pt')

# 주어진 라인의 출력 반환
def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

'''
# 예시들 중에 어떤 것이 정확하게 예측되었는지 기록
for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

# 모든 행을 합계로 나누어 정규화
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 도식 설정
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 축 설정
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# 모든 tick에서 레이블 지정
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
'''

#몇 가지 테스트
def predict(input_line,n_predictions=1):
    print('\n>%s'%input_line)
    # 테스트데이터는 오차조정에 사용하지 않음
    with torch.no_grad():
        output=evaluate(lineToTensor(input_line))

        #top N categories가져오기
        topv,topi=output.topk(n_predictions,1,True)
        predictions=[]
        for i in range(n_predictions):
            value=topv[0][i].item()
            category_index=topi[0][i].item()
            print('(%.2f)%s'%(value,all_categories[category_index]))
            predictions.append([value,all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
