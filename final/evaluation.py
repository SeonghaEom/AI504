import zipfile
import nltk

STUDENT_ID = 20223389

with zipfile.ZipFile(f'{STUDENT_ID}.zip') as zf:
    zf.extractall(".")
with open(f'{STUDENT_ID}/gen.txt') as genf:
    gen = [x.strip().strip(".").split() for x in genf.read().splitlines()]
with open(f'{STUDENT_ID}/gt.txt') as gtf:
    gt = [[x.strip().strip(".").split()] for x in gtf.read().splitlines()]

print("Your BLEU score:", nltk.translate.bleu_score.corpus_bleu(gt, gen))
