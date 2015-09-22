import numpy as np
from scipy.optimize import minimize as mymin
from my_maxent_test import MyMaxEnt
import feature_functions as f_fn
import nltk


def extract_entity_names(t,sent):
    l=[]
    for n in t:
        if isinstance(n, nltk.tree.Tree):
            l.append((str(n).split("/")[0].split(" ")[1],n.label()))
        else:
            l.append((n[0],'OTHER'))
    tagged_nltk[sent]=l


def max_ent_test():
    '''
        Our test function. Stores result as {sentence:[(word,tag),(word,tag)..]}
    '''
    u='*'
    v='*'
    for i in range(len(tokenized_sentences)):
        l=[]
        for j in range(len(tokenized_sentences[i])):
            if((j-2)<0 and (j-1)<0):
                hist_tuple = ('*','*',tuple(tokenized_sentences[i]),0)
                ner_tagger.fvectors[hist_tuple] = {}
                for tag in ner_tagger.tags:
                    ner_tagger.fvectors[hist_tuple][tag] = np.array([fun(hist_tuple,tag) for fun in feature_fn_list])

                u=ner_tagger.classify(hist_tuple)
                print(u)
                l.append((tokenized_sentences[i][j],u))

            elif ((j-2)<0 and (j-1)==0):
                hist_tuple = ('*',u,tuple(tokenized_sentences[i]),1)
                ner_tagger.fvectors[hist_tuple] = {}

                for tag in ner_tagger.tags:
                    ner_tagger.fvectors[hist_tuple][tag] = np.array([fun(hist_tuple,tag) for fun in feature_fn_list])
                v=ner_tagger.classify(hist_tuple)
                print(v)
                l.append((tokenized_sentences[i][j],v))

            else:
                hist_tuple=(u,v,tuple(tokenized_sentences[i]),j)

                ner_tagger.fvectors[hist_tuple] = {}
                for tag in ner_tagger.tags:
                    ner_tagger.fvectors[hist_tuple][tag] = np.array([fun(hist_tuple,tag) for fun in feature_fn_list])
                prev=ner_tagger.classify(hist_tuple)
                u=v
                v=prev
                print(u,v,prev)
                l.append((tokenized_sentences[i][j],prev))
            my_tagged_sent[sentences[i]]=l


def nltk_tagger():
    '''
    	NTLK Tagger. Stores result as {sentence:[(word,tag),(word,tag)..]}
    '''
    tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
    chunked_sentences = []
    for i in tagged_sentences:
        chunked_sentences.append(nltk.chunk.ne_chunk(i))
    ctr=0
    for tree in chunked_sentences:
        extract_entity_names(tree,sentences[ctr])
        ctr+=1

def test_accuracy():
	count = 0
	for i in range(len(my_tagged_sent.values())):
		for j in range(len(my_tagged_sent.values()[i])):
			if my_tagged_sent.values()[i][j][1] == tagged_nltk.values()[i][j][1]:
				count += 1

	print 'Accuracy of our Model wrt NTLK Tagger : ',float(count)/sum([len(i) for i in my_tagged_sent.values()]) * 100,'%'



if __name__ == "__main__":
    tagged_nltk=dict()  # Result for our tagging
    my_tagged_sent=dict() # Result of NLTK tagging

    hist_list = eval(open('history.txt').read())
    feature_fn_list = [f_fn.f1,f_fn.f2,f_fn.f3,f_fn.f4,f_fn.f5,f_fn.f6,f_fn.f7,f_fn.f8,f_fn.f9,f_fn.f10,f_fn.f11]

    ner_tagger = MyMaxEnt(hist_list,feature_fn_list)
    ner_tagger.train()
    ner_tagger.save("model_wo_gradient.pickle")
    #ner_tagger = ner_tagger.load("model_wo_gradient.pickle")
    tokenized_sentences = eval(open("test_sentences.txt").read())
    sentences = []
    for s in tokenized_sentences:
        sentences.append(" ".join(s))
    #test_sample = sentences    
#test_sample=" ".join(['Steve','Jobs','told','that','the','Apple', 'iPhone', 'is', 'a', 'new', 'addition', 'to', 'the', 'Apple','family','priced','at','$','41,000'])
    #test_sample = str(raw_input('Enter a sentence : '))
    #sentences=nltk.sent_tokenize(test_sample)
    #tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
	
    nltk_tagger()
    max_ent_test()

    print '###################################################'
    print " NEs Tagged by our Model : "
    #for i in my_tagged_sent[test_sample]:
    #	print i[0],' => ',i[1]
    print '###################################################'
    print "NEs Tagged by NLTK:"
    #for i in tagged_nltk[test_sample]:
    #	print i[0],' => ',i[1]
    print '###################################################'
    test_accuracy()
    print '###################################################'
