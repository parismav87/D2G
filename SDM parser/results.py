import csv
import matplotlib.pyplot as plt
import sys
import os
from sklearn.cluster import MeanShift, KMeans, SpectralClustering
import numpy as np


def clustering(arr1, arr2, label1, label2):
	X = np.vstack((arr1, arr2)).T
	# print X
	kmeans = KMeans(n_clusters = 2).fit_predict(X, y=None)
	plt.scatter(X[:,0], X[:,1], c=kmeans)
	plt.xlabel(label1)
	plt.ylabel(label2)	
	plt.show()


class Dilemma(object):
	def __init__(self):
		self.gameId = ""
		self.dilemmaId = ""
		self.timeOpen = 0
		self.answerType = ""
		self.timeToAnswer = 0
		self.timesOpened = 0
		self.numberInfosRead = 0
		self.numberInfosTotal = 5
		self.adviceRequested = False
		self.empathyFeedback = 0
		self.directInfos = 0
		self.indirectInfos = 0
		self.numberAskedInfos = 0
		self.sumAdvisorMood = 0
		self.game = game
		self.initTime = 0
		self.endTime = 0
		self.initIndex = 0
		self.endIndex = 0
		self.avgHR = 0
		self.stdHR = 0
		self.minHR = 0
		self.maxHR = 0
		self.diffHR = 0
		self.initHR = 0
		self.endHR = 0
		self.shiftHR = 0
		self.timestampMaxHR = 0
		self.timestampMinHR = 0
		self.diffTimestampHR = 0
		self.hrv = 0
		self.minSC = 0
		self.maxSC = 0
		self.diffSC = 0
		self.initSC = 0
		self.endSC = 0
		self.shiftSC = 0
		self.timestampMaxSC = 0
		self.timestampMinSC = 0
		self.diffTimestampSC = 0
		self.avgSC = 0
		self.stdSC = 0


class Game:	
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
		self.shiftSC = float(shiftSC)
		self.timestampMaxSC = float(timestampMaxSC)
		self.timestampMinSC = float(timestampMinSC)
		self.diffTimestampSC = float(diffTimestampSC)
		self.avgSC = float(avgSC)
		self.stdSC = float(stdSC)
		self.version = 0
		self.sumAdvisorMood = 0
		self.averageInfoOpen = 0
		self.stdInfoOpen = 0 
		self.averageTimeToAnswerDilemma = 0 
		self.stdTimeToAnswerDilemma = 0
		self.averageTimeDilemmaOpen = 0
		self.stdTimeDilemmaOpen = 0
		self.sumPosFeedback = 0
		self.sumNegFeedback = 0
		self.sumNeuFeedback = 0
		self.sumAdviceRequested = 0
		self.sumInfosRead = 0
		self.sumInfos = 0
		self.avgTimesDilemmaOpened = 0
		self.stdTimesDilemmaOpened = 0
		self.sumDirectInfo = 0
		self.sumIndirectInfo = 0
		self.dilemmaArray = []



games = []
arr1 = []
arr2 = []

with open("output/sensorStats.csv") as f:
	reader = csv.reader(f)
	for k,v in enumerate(reader):
		if k>0:
			r = Game(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],v[9],v[10],v[11],v[12],v[13],v[14],v[15],v[16],v[17],v[18],v[19],v[20],v[21],v[22],v[23])
			games.append(r)
			# print r.avgHR
 
with open("output/stats.csv") as f:
	reader = csv.reader(f)
	for k,v in enumerate(reader):
		if k>0:
			gameid = v[0]
			for game in games:
				if gameid == game.gameid:
					game.version = int(v[1])
					game.sumAdvisorMood = int(v[2])
					game.averageInfoOpen = float(v[3])
					game.stdInfoOpen = float(v[4])
					game.averageTimeToAnswerDilemma = float(v[5])
					game.stdTimeToAnswerDilemma = float(v[6])
					game.averageTimeDilemmaOpen = float(v[7])
					game.stdTimeDilemmaOpen = float(v[8])
					game.sumPosFeedback = int(v[9])
					game.sumNegFeedback = int(v[10])
					game.sumNeuFeedback = int(v[11])
					game.sumAdviceRequested = int(v[12])
					game.sumInfosRead = int(v[13])
					game.sumInfos = int(v[14])
					game.avgTimesDilemmaOpened = float(v[15])
					game.stdTimesDilemmaOpened = float(v[16])
					game.sumDirectInfo = int(v[17])
					game.sumIndirectInfo = int(v[18])
 


d1 = "output"
listdirs = []
for f1 in os.listdir(d1):
	# print f1
	if ".csv" not in f1:
		d2 = os.path.join(d1,f1)
		listdirs.append(d2)

for f2 in listdirs:
	for f3 in os.listdir(f2):
		filepath = os.path.join(f2,f3)
		# print filepath
		with open(filepath) as f:
			reader = csv.reader(f)
			for k,v in enumerate(reader):
				if k>0:
					gameid = v[0]
					for game in games:
						if gameid == game.gameid:
							d = Dilemma()
							d.gameId = v[1]
							d.dilemmaId = v[2]
							d.initIndex = v[3]
							d.timeOpen = v[4]
							d.answerType = v[5]
							d.timeToAnswer = v[6]
							d.timesOpened = v[7]
							d.numberInfosTotal = v[8]
							d.numberInfosRead = v[9]
							d.adviceRequested = v[10]
							d.empathyFeedback = v[11]
							d.directInfos = v[12]
							d.indirectInfos = v[13]
							d.avgHR = v[14]
							d.stdHR = v[15]
							d.minHR = v[16]
							d.maxHR = v[17]
							d.diffHR = v[18]
							d.initHR = v[19]
							d.endHR = v[20]
							d.shiftHR = v[21]
							d.timestampMaxHR = v[22]
							d.timestampMinHR = v[23]
							d.diffTimestampHR = v[24]
							d.hrv = v[25]
							d.avgSC = v[26]
							d.stdSC = v[27]
							d.minSC = v[28]
							d.maxSC = v[29]
							d.diffSC = v[30]
							d.initSC = v[31]
							d.endSC = v[32]
							d.shiftSC = v[33]
							d.timestampMaxSC = v[34]
							d.timestampMinSC = v[35]
							d.diffTimestampSC = v[36]
							game.dilemmaArray.append(d)


for k in games:
	# print k.diffSC
	arr1.append(k.timestampMaxSC)
	arr2.append(k.shiftSC)

clustering(arr1, arr2, "timestamp max sc", "average time to answer dilemma")