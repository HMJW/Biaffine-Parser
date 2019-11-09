# 挑数据，只保留词数[8-40]的句子，根据概率分为3等份，100:150:250
# train 120w 挑4000句；800：1200 : 2000
# test-all  挑2000句；400 : 600 : 1000
import random

in_file = ["../data/NMT/nist-split/nist02-codt-conllx.in", "../data/NMT/nist-split/nist03-codt-conllx.in","../data/NMT/nist-split/nist04-codt-conllx.in","../data/NMT/nist-split/nist05-codt-conllx.in","../data/NMT/nist-split/nist06-codt-conllx.in"]
out_file = "./test.out"
samples = [1000, 600, 400]

sen, sens = [], []
for filename in in_file:
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line == "\n":
                if 8 <= len(sen) <= 40:
                    sens.append(sen)
                sen = []
            else:
                sen.append(line.strip().split())

print(len(sens))
def key(sen):
    _, _, _, _, _, _, _, _, _, probs = zip(*sen)
    p = list(map(float, probs))
    return sum(p)/len(p)

sorted_sens = sorted(sens, key=key)
size = len(sorted_sens)
chunk_size = size // 3
chunks = [sorted_sens[i * chunk_size : i * chunk_size + chunk_size] for i in range(3)]
result = []
for c, s in zip(chunks, samples):
    result += random.sample(c, s)
random.shuffle(result)

with open(out_file, "w", encoding="utf-8") as f:
    for sen in result:
        for line in sen:
            f.write("\t".join(line))
            f.write("\n")
        f.write("\n")
