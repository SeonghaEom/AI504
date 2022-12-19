from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import nltk
from tqdm import tqdm
import torch
device = torch.device('cuda')

mname = "facebook/wmt19-de-en"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname, device_map="auto", load_in_8bit=True)

generated = []
with open('../final/gen.txt', 'w') as genf:

    with open('.data/multi30k/test2016.de') as testfile:
        inputs = [x for x in testfile.read().splitlines()]
        for input in tqdm(inputs):
    # input = "Maschinelles Lernen ist großartig, oder?"
            input_ids = tokenizer.encode(input, return_tensors="pt")
            outputs = model.generate(input_ids)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print(decoded) # Machine learning is great, isn't it?
            generated.append(decoded)
            print(decoded)
            genf.write(decoded)
            genf.write('\n')
# with open('../final/gen.txt') as genf:
#     gen = [x.strip().strip(".").split() for x in genf.read().splitlines()]
gen = [x.strip().strip(".").split() for x in generated]
with open('../final/gt.txt') as gtf:
    gt = [[x.strip().strip(".").split()] for x in gtf.read().splitlines()]

print("Your BLEU score:", nltk.translate.bleu_score.corpus_bleu(gt, gen))