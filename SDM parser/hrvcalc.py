import csv

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import signal

def getMovAvg(dataList, window):
	movAvg = []
	avg = np.mean(dataList)
	cumSum = [0]
	
	for k,v in enumerate(dataList,1):
		cumSum.append(cumSum[k-1] + v) 
		if k >= window:
			m = (cumSum[k] - cumSum[k-window]) / window
			augmented = m  #to avoid peak detection in 2nd heart contraction
			movAvg.append(augmented)
		else:
			movAvg.append(0)
	return movAvg

def getPeakDetection(dataList, movAvg):
	window = []
	peakList = []
	for k,v in enumerate(dataList):
		m = movAvg[k]
		if v <= m and len(window)<1:
			pass
		elif v > m :
			window.append(v)
		else :
			maximum = max(window)
			beatPosition = k - len(window) + window.index(max(window))
			peakList.append(beatPosition)
			window = []
	if len(window) > 1:
		beatPosition = len(dataList) - len(window) + window.index(max(window))
		peakList.append(beatPosition)
	return peakList


def getIntervals(peakList):
	intervals = []
	for k,v in enumerate(peakList):
		if k < len(peakList)-1:
			interval = peakList[k+1] - v  #interval is already in ms, sampling freq is 1000
			dist = (interval* 1000 / 1024) 
			intervals.append(dist)			
	intervals.append(((peakList[len(peakList)-1] - peakList[len(peakList)-2])*1000/1024)) #last element
	return intervals


def getBpm(intervals, peakList):
	bpms = []
	start = 0
	for k,v in enumerate(intervals):
		if v>0:
			for i in range(start, peakList[k]):
				bpms.append(60000/v) #transform to beats per sec
			start = peakList[k]+1
	return bpms

def getIntervalDiffs(intervals):
	intervalDiffs = []
	for k,v in enumerate(intervals):
		if k>0:
			intervalDiffs.append(abs(intervals[k] - intervals[k-1]))
	return intervalDiffs

def getIntervalSqdiffs(intervals):
	intervalSqdiffs = []
	for k,v in enumerate(intervals):
		if k>0:
			intervalSqdiffs.append(math.pow(intervals[k] - intervals[k-1], 2))
	return intervalSqdiffs

def getRmssd(intervalSqdiffs):
	return np.sqrt(np.mean(intervalSqdiffs))

def getButterWorth(data, cutoff, order):
	nyq = 0.5 * 1024 #Nyquist frequeny is half the sampling frequency
	normal_cutoff = cutoff / nyq
	b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
	y = signal.lfilter(b, a, data)
	return y

def getInGameHR(game):
	for i,v in enumerate(game.rawHR):
		if game.timestamps[i] >= game.gamestartUTC and game.timestamps[i] <= game.gameendUTC and v>0: #filter out <=0 values
			game.rawHRInGame.append(game.rawHR[i])

def getSCStats(game):
	firstMeasurement = False
	lastMeasurement = False
	minSC = 9999
	maxSC = 0
	timestampMaxSC = 0
	timestampMinSC = 0

	for i,v in enumerate(game.scSignal):
		if game.timestamps[i] >= game.gamestartUTC and game.timestamps[i] <= game.gameendUTC and v>0: #filter out <0 values
			game.scInGame.append(v) # add hr value to ingame HR
			if not firstMeasurement:
				firstMeasurement = True
				game.initSC = v
			if v> maxSC:
				maxSC = v
				timestampMaxSC = i
			if v< minSC:
				minSC = v
				timestampMinSC = i
		elif game.timestamps[i]> game.gameendUTC and not lastMeasurement:
			lastMeasurement = True
			game.endSC = game.scSignal[i-1]

	game.avgSC = np.average(game.scInGame)
	game.stdSC = np.std(game.scInGame)

	for s in game.scInGame:
		s = s - game.avgSC

	game.maxSC = maxSC - game.avgSC
	game.minSC = minSC - game.avgSC
	game.timestampMaxSC = float(timestampMaxSC)/len(game.timestamps)
	game.timestampMinSC = float(timestampMinSC)/len(game.timestamps)
	game.diffSC = maxSC - minSC
	game.diffTimestampSC = math.fabs(game.timestampMaxSC - game.timestampMinSC)


def getHRStats(game, bpms):

	minHR = 9999
	maxHR = 0
	timestampMaxHR = 0
	timestampMinHR = 0
	initHR = bpms[0]
	endHR = bpms[len(bpms)-1]

	for i,v in enumerate(bpms):
		if v> maxHR:
			maxHR = v
			timestampMaxHR = i
		if v< minHR:
			minHR = v
			timestampMinHR = i
		game.hrInGame.append(v) # add hr value to ingame HR
		

	game.avgHR = np.average(game.hrInGame)
	game.stdHR = np.std(game.hrInGame)

	# plt.plot(game.hrInGame)
 #   	plt.show()

	for h in game.hrInGame:
		game.hrBaseline = h-game.avgHR


	game.maxHR = maxHR - game.avgHR
	game.minHR = minHR - game.avgHR
	game.initHR = initHR - game.avgHR
	game.endHR = endHR - game.avgHR
	game.timestampMaxHR = float(timestampMaxHR)/len(game.timestamps)
	game.timestampMinHR = float(timestampMinHR)/len(game.timestamps)
	game.diffHR = maxHR - minHR
	game.diffTimestampHR = math.fabs(game.timestampMaxHR - game.timestampMinHR)
	

	print "---------------------"
	print game.maxHR
	print game.minHR
	print game.initHR
	print game.endHR
	print game.timestampMaxHR
	print game.timestampMinHR
	print game.diffHR
	print game.diffTimestampHR
	print game.avgHR
	print game.stdHR
	print game.hrv
	print "---------------------"

def run(game):
	rawHR = game.rawHR
	getInGameHR(game)
	buff = []
	for r in game.rawHRInGame:
		buff.append(math.pow(r,3))



	filteredHR = getButterWorth(buff, 1.5 , 2)
	# filteredHR = rawHR
	mov_avg = getMovAvg(filteredHR, 768) #0.75*1024
	peakList = getPeakDetection(filteredHR, mov_avg)
	ybeat = [filteredHR[x] for x in peakList]
	intervals = getIntervals(peakList)
	sqdiffs = getIntervalSqdiffs(intervals)
	rmssd = getRmssd(sqdiffs)
	bpms = getBpm(intervals, peakList)


	game.hrv = rmssd

	getHRStats(game,bpms)
	getSCStats(game)

	# print "HRV: ", rmssd
	# print "avg BPM: ", np.mean(bpms)
	# plt.ion()
	if rmssd>190:
		# plt.plot(buff)
		plt.plot(filteredHR)
		plt.plot(mov_avg)
		plt.scatter(peakList, ybeat, color="red")
		# plt.plot(bpms)
		plt.show()
	
	# print len(filteredHR)
	# print len(bpms)