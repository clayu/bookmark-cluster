import numpy as np
import re
import scam_dist as scam
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from os import listdir
from os.path import isfile, join
import math

dir = "C:\\Users\\Clay\\Dropbox\\DocumentMap\\scripts\\docs\\"
outDir = "C:\\Users\\Clay\\Dropbox\\DocumentMap\\scripts\\out\\"

def getFiles(mypath):
	return [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

def createOutput(files,arr):
	output="{"
	output+='"nodes": ['
	nodes = []
	for f in files:
		nodes.append('{"name":"'+f+'","group":1}')
	output+=','.join(nodes)
	output+='],"links":['	
	links = []
	for i in range(len(arr)):
		for j in range(len(arr[i])):
			if (arr[i][j]<.25 and arr[i][j]>=0):
				links.append('{"source":'+str(i)+',"target":'+str(j)+',"value":1,"strength":'+str(arr[i][j])+'}')
	output+=','.join(links)	
	output+="]"	
	output+="}"
	
	output = output.replace(":nan",":1")
	
	fout = open(outDir+"readme.json", 'wb')
	fout.write(output)
	

def main():
	docs = []
	cats = []  

	files = getFiles(dir)
	n=len(files)
	#files = ["C:\\Users\\Clay\\Dropbox\\DocumentMap\\scripts\\docs\\d1.txt", "C:\\Users\\Clay\\Dropbox\\DocumentMap\\scripts\\docs\\d2.txt", "C:\\Users\\Clay\\Dropbox\\DocumentMap\\scripts\\docs\\d3.txt"]
	for ff in files:
		file=dir+ff
		f = open(file, 'rb')
		body = re.sub("\\s+", " ", " ".join(f.readlines()))
		f.close()
		docs.append(body)
		cats.append("X")
	pipeline = Pipeline([
		("vect", CountVectorizer(min_df=0.0,max_df=0.1)),
		("tfidf", TfidfTransformer(use_idf=True,norm="l2"))])
	tdMatrix = pipeline.fit_transform(docs, cats)
	testDocs = []
	print cats
	for i in range(0, tdMatrix.shape[0]):
		testDocs.append(np.asarray(tdMatrix[i, :].todense()).reshape(-1))
	
	arr=np.zeros((n,n))
	
	for i in range(0,n):
		for j in range(i,n):
			dist = scam.scam_distance(testDocs[i], testDocs[j])
			if (math.isnan(dist)):
				dist=1
			arr[i][j]=dist
			arr[j][i]=arr[i][j]
	print ""
	print('\n'.join(['{:12}'.format(files[r])+"  "+''.join(['{:6.2f}'.format(item) for item in arr[r]]) for r in range(len(arr))]))
	print ""
	createOutput(files,arr)

if __name__ == "__main__":
	main()