import csv
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import MeanShift, KMeans, SpectralClustering
import numpy as np


def clustering(arr1, arr2, label1, label2):
	X = np.vstack((arr1, arr2)).T
	# print X
	kmeans = SpectralClustering(n_clusters = 5).fit_predict(X, y=None)
	plt.scatter(X[:,0], X[:,1], c=kmeans)
	plt.xlabel(label1)
	plt.ylabel(label2)	
	plt.show()

class result:	
	def __init__(self, gameid, maxHR, minHR, initHR, endHR,	shiftHR, timestampMaxHR, timestampMinHR, diffHR, diffTimestampHR, avgHR, stdHR, 
		hrv, maxSC,	minSC, initSC, endSC, shiftSC, timestampMaxSC, timestampMinSC, diffSC, diffTimestampSC, avgSC, stdSC):
		self.gameid = gameid
		self.avgHR = float(avgHR)
		self.stdHR = float(stdHR)
		self.minHR = float(minHR)
		self.maxHR = float(maxHR)
		self.diffHR = float(diffHR)
		self.initHR = float(initHR)
		self.endHR = float(endHR)
		self.shiftHR = float(shiftHR)
		self.timestampMaxHR = float(timestampMaxHR)
		self.timestampMinHR = float(timestampMinHR)
		self.diffTimestampHR = float(diffTimestampHR)
		self.hrv = float(hrv)
		self.minSC = float(minSC)
		self.maxSC = float(maxSC)
		self.diffSC = float(diffSC)
		self.initSC = float(initSC)
		self.endSC = float(endSC)
		self.shiftSC = float(shiftSC) * -1
		self.timestampMaxSC = float(timestampMaxSC)
		self.timestampMinSC = float(timestampMinSC)
		self.diffTimestampSC = float(diffTimestampSC)
		self.avgSC = float(avgSC)
		self.stdSC = float(stdSC)



results = []
arr1 = []
arr2 = []

with open("output/sensorStats.csv") as f:
	reader = csv.reader(f)
	for k,v in enumerate(reader):
		if k>0:
			r = result(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19],v[20],v[21],v[22],v[23])
			results.append(r)
			# print r.avgHR
 

for k in results:
	# print k.diffSC
	arr1.append(k.timestampMaxSC)
	arr2.append(k.diffSC)

clustering(arr1, arr2, "max - min SC", "end - init SC")