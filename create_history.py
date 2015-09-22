import nltk

def create_history(tag_dict):
    """Method to create history tuples"""
    history = []
    for i in tag_dict.keys():
        for j in range(len(tag_dict[i])):
            if((j-2)<0):
                u = '*'
            else:
                u = tag_dict[i][j-2][1]

            if((j-1)<0):
                v = '*'
            else:
                v = tag_dict[i][j-1][1]

            x = (u,v,tuple(nltk.word_tokenize(i)),j)
            history.append(x)
    return history

tagged_history=dict()

def extract_entity_names(t,sent):
    l=[]
    for n in t:
        if isinstance(n, nltk.tree.Tree):
            l.append((str(n).split("/")[0].split(" ")[1],n.label()))
        else:
            l.append((n[0],'OTHER'))
    tagged_history[sent]=l

with open('apple-iphone-5-first-look.txt', 'r') as f:
    sample = f.read()
     
sentences = nltk.sent_tokenize(sample)
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
#print tagged_sentences
chunked_sentences = []
for i in tagged_sentences:
    chunked_sentences.append(nltk.chunk.ne_chunk(i))
#print chunked_sentences


entity_names = []
tags = dict()
ctr=0
for tree in chunked_sentences:
    extract_entity_names(tree,sentences[ctr])
    ctr+=1
# print(tagged_history)
h = create_history(tagged_history)
f = open("history.txt","w")
f.write(str(h))
f.close()