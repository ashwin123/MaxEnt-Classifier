import numpy as np
from scipy.optimize import minimize as mymin
from my_maxent import MyMaxEnt
import feature_functions as f_fn

hist_list = eval(open('history.txt').read())
feature_fn_list = [f_fn.f1,f_fn.f2,f_fn.f3,f_fn.f4,f_fn.f5,f_fn.f6,f_fn.f7,f_fn.f8,f_fn.f9,f_fn.f10]

ner_tagger = MyMaxEnt(hist_list,feature_fn_list)
ner_tagger.train()

ner_tagger.save("ner_iphone5.txt")


