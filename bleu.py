from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
 
#不需要分词
src='today is sunday'
tgt='today is good'
 
smooth = SmoothingFunction()
score = sentence_bleu([src], tgt, smoothing_function=smooth.method1)
 
print(score)

