import nltk,re
def f1(h):
	'''
		Check if first is capital
	'''
	if(h[2][h[3]].istitle()):
		return 1
	else:
		return 0
		
def f2(h):
	'''
		use dictionary for organization enumeration
	'''
	dic = ["alcatel","amazon","apple","asus","blackberry","byond","coolpad","gionee","google","hp","honor","htc","huawei","iball","infocus","intex","jolla","karbonn","lava","lenovo","lg","micromax","microsoft","motorola","nokia","oneplus","samsung","oppo","panasonic","penta","phicomm","philips","sony","spice","vivo","wickedleak","xiaomi","xolo","yu","zte","kingston","pebble","strontium","sunstrike","sandisk"]
	# print h[2][h[3]]
	if(h[2][h[3]].lower() in dic):
		return 1
	else:
		return 0

def f3(h):
	'''
		Location if previous is "in" or "at"
	'''
	if(h[2][h[3]-1] in ["in","at"]):
		return 1
	else:
		return 0

def f4(h):
	'''
		Money regex
	'''
	dic = []
	print re.match(r'\d+',h[2][h[3]]), h[2][h[3]]
	if(re.match(r'\d+',h[2][h[3]]) and(h[2][h[3]-1] in ["$","Rs"] or h[2][h[3]-2] in ["$","Rs"])):
		return 1
	else:
		return 0
		
def f5(h):
	'''
		Organization followed by inc or corp
	'''
	print h[2][h[3]+1]
	if(h[2][h[3]+1] in ["inc.","corp","inc","corp."]):
		return 1
	else:
		return 0	
		
def f6(h):
	if(re.search('^(([0-1]?[0-9])|([2][0-3])):([0-5]?[0-9])(:([0-5]?[0-9]))?$',h[2][h[3]]) or re.search('^((0?[13578]|10|12)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[01]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1}))|(0?[2469]|11)(-|\/)(([1-9])|(0[1-9])|([12])([0-9]?)|(3[0]?))(-|\/)((19)([2-9])(\d{1})|(20)([01])(\d{1})|([8901])(\d{1})))$',h[2][h[3]])):
		return 1
	else:
		return 0

def f7(h):
	'''
		Organization followed by product model
	'''
	for i in range(1,4):
		if(re.match(r'\d*\[a-zA-Z]+\d+',h[2][h[3]+i])):
			return 1
	else:
		return 0

def f8(h):
	'''
		City, State/Country eg: Bangalore, India
	'''
	if(h[0]=="LOCATION" and h[2][h[3]-1]==","):
		return 1
	else:
		return 0
		
def f9(h):
	'''
		Person if next word is said,rebutted .....
	'''
	if(h[2][h[3]+1] in ["said","rebutted","told"]):
		return 1
	else:
		return 0	
		
def f10(sentence):
	'''
		if in a window of 5 there is a stem with cost or price
	'''
	import re,nltk
	porter_stemmer = nltk.PorterStemmer()
	l1 = nltk.word_tokenize(sentence)
	words=[]
	for i in l1:
		words.append(porter_stemmer.stem(i))
	for i in range(0,len(words)-5,5):
		if(("cost" in words or "price" in words)):
			for w in words:
				if(re.search(r"\d+",w)):
					return 1
	return 0

if __name__=="__main__":
	print f8(("LOCATION","t1",nltk.word_tokenize("Hello there Ashwin is in Bangalore, India Microsoft 50 for Rs. 100."),7))