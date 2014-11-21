import nltk, re, pprint
from nltk import word_tokenize
import urllib2
from bs4 import BeautifulSoup

def train(fnin):
  docs = []
  cats = []
  fin = open(fnin, 'rb')
  for line in fin:
    id, category, body = line.strip().split("\t")
    docs.append(body)
    cats.append(category)
  fin.close()
  pipeline = Pipeline([
    ("vect", CountVectorizer(min_df=0, stop_words="english")),
    ("tfidf", TfidfTransformer(use_idf=False))])
  tdMatrix = pipeline.fit_transform(docs, cats)
  return tdMatrix, cats



url = "http://www.nltk.org/book/ch03.html"
response = urllib2.urlopen(url)
rawUnstripped = response.read().decode('utf8')
soup = BeautifulSoup(rawUnstripped)
for elem in soup.findAll(['script', 'style']):
	elem.extract()
raw=soup.get_text()
tokens =[x for x in word_tokenize(raw) if len(x)>1]
text = nltk.Text(tokens)
print text.collocations()