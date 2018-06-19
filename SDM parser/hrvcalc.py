import csv

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy import signal

def getMovAvg(dataList, window, aug):
	movAvg = []
	avg = np.mean(dataList)
	cumSum = [0]
	
	for k,v in enumerate(dataList,1):
		cumSum.append(cumSum[k-1] + v) 
		if k >= window:
			m = (cumSum[k] - cumSum[k-window]) / window
			augmented = m + ((m*aug)/100) #to avoid peak detection in 2nd heart contraction
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
			interval = peakList[k+1] - v  
			dist = (interval* 1000 / 1024) 
			intervals.append(dist)			
	intervals.append(((peakList[len(peakList)-1] - peakList[len(peakList)-2])*1000/1024)) #last element
	return intervals


def getBpm(intervals, peakList, game):
	bpms = []
	start = 0
	for k,v in enumerate(intervals):
		if v>0:
			for i in range(start, peakList[k]):
				bpms.append(60000/v) #transform to beats per sec
			start = peakList[k]+1
	if len(bpms)>0:
		lastBpm = bpms[len(bpms)-1]
		for q in range(len(bpms), len(game.rawHRInGame)):
			bpms.append(lastBpm)
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
			game.inGameTimestamps.append(game.timestamps[i])

def getSCStats(game):
	firstMeasurement = False
	lastMeasurement = False
	minSC = 9999
	maxSC = 0
	timestampMaxSC = 0
	timestampMinSC = 0

	filt = signal.medfilt(game.scSignal, 513)
	filt2 = scipy.ndimage.filters.convolve(filt, np.full((2048), 1.0/2048))
	game.scSignalFiltered = filt2



	for i,v in enumerate(game.scSignalFiltered):
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
			game.endSC = game.scSignalFiltered[i-1]

	game.avgSC = np.average(game.scInGame)
	game.stdSC = np.std(game.scInGame)

	for k,v in enumerate(game.scInGame):
		game.scBaseline.append(v - game.avgSC)

	for i,v in enumerate(game.scInGame):
		if i<len(game.scInGame)-2:
			game.firstDiffSC.append(game.scInGame[i+1] - game.scInGame[i])

	game.maxSC = maxSC - game.avgSC
	game.minSC = minSC - game.avgSC
	game.timestampMaxSC = float(timestampMaxSC)/len(game.timestamps)
	game.timestampMinSC = float(timestampMinSC)/len(game.timestamps)
	game.initSC = game.initSC - game.avgSC
	game.endSC = game.endSC - game.avgSC
	game.diffSC = maxSC - minSC
	game.diffTimestampSC = math.fabs(game.timestampMaxSC - game.timestampMinSC)
	game.shiftSC = game.endSC - game.initSC


	# for d in game.dilemmaArray:
	# 	print d.dilemmaId, "  ", d.endTime
	# print '---'
	

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
		if i<len(bpms)-2:
			game.firstDiffHR.append(bpms[i+1] - bpms[i])

	game.avgHR = np.average(game.hrInGame)
	game.stdHR = np.std(game.hrInGame)

	for k,v in enumerate(game.hrInGame):
		game.hrBaseline.append(v - game.avgHR)
	# plt.plot(game.hrInGame)
 #   	plt.show()

	game.maxHR = maxHR - game.avgHR
	game.minHR = minHR - game.avgHR
	game.initHR = initHR - game.avgHR
	game.endHR = endHR - game.avgHR
	game.shiftHR = game.endHR - game.initHR
	game.timestampMaxHR = float(timestampMaxHR)/len(game.timestamps)
	game.timestampMinHR = float(timestampMinHR)/len(game.timestamps)
	game.diffHR = maxHR - minHR
	game.diffTimestampHR = math.fabs(game.timestampMaxHR - game.timestampMinHR)
	

	# print "---------------------"
	# print game.maxHR
	# print game.minHR
	# print game.initHR
	# print game.endHR
	# print game.shiftHR
	# print game.timestampMaxHR
	# print game.timestampMinHR
	# print game.diffHR
	# print game.diffTimestampHR
	# print game.avgHR
	# print game.stdHR
	# print game.hrv
	# print "---------------------"




def getDilemmaStats(game):

	for d in game.dilemmaArray:
		dilemmaHR = []
		dilemmaSC = []

		for i in range(d.initIndex, d.endIndex):
			dilemmaHR.append(game.hrInGame[i])
			dilemmaSC.append(game.scInGame[i])

		d.avgHR = np.mean(dilemmaHR)
		d.stdHR = np.std(dilemmaHR)
		d.minHR = np.amin(dilemmaHR) - d.avgHR
		d.maxHR = np.amax(dilemmaHR) - d.avgHR
		d.diffHR = d.maxHR - d.minHR
		d.initHR = dilemmaHR[0] - d.avgHR
		d.endHR = dilemmaHR[len(dilemmaHR)-1] - d.avgHR
		d.shiftHR = d.endHR - d.initHR
		d.timestampMaxHR = float(np.argmax(dilemmaHR))/len(dilemmaHR)
		d.timestampMinHR = float(np.argmin(dilemmaHR))/len(dilemmaHR)
		d.diffTimestampHR = d.timestampMaxHR - d.timestampMinHR

		d.avgSC = np.mean(dilemmaSC)
		d.stdSC = np.std(dilemmaSC)
		d.minSC = np.amin(dilemmaSC) - d.avgSC
		d.maxSC = np.amax(dilemmaSC) - d.avgSC
		d.diffSC = d.maxSC - d.minSC
		d.initSC = dilemmaSC[0] - d.avgSC
		d.endSC = dilemmaSC[len(dilemmaSC)-1] - d.avgSC
		d.shiftSC = d.endSC - d.initSC
		d.timestampMaxSC = float(np.argmax(dilemmaSC))/len(dilemmaSC)
		d.timestampMinSC = float(np.argmin(dilemmaSC))/len(dilemmaSC)
		d.diffTimestampSC = d.timestampMaxSC - d.timestampMinSC


		print d.avgHR, " avgHR "
		print d.stdHR, " stdHR "
		print d.minHR, " minHR "
		print d.maxHR, " maxHR "
		print d.diffHR, " diffHR "
		print d.initHR, " initHR "
		print d.endHR, " endHR "
		print d.shiftHR, " shiftHR "
		print d.timestampMaxHR, " timestampMaxHR "
		print d.timestampMinHR, " timestampMinHR "
		print d.diffTimestampHR, " diffTimestampHR "
		
		print d.avgSC, " avgSC "
		print d.stdSC, " stdSC "
		print d.minSC, " minSC "
		print d.maxSC, " maxSC "
		print d.diffSC, " diffSC "
		print d.initSC, " initSC "
		print d.endSC, " endSC "
		print d.shiftSC, " shiftSC "
		print d.timestampMaxSC, " timestampMaxSC "
		print d.timestampMinSC, " timestampMinSC "
		print d.diffTimestampSC, " diffTimestampSC "


		print "-----"
		
	

def detoneSC(game):
	finished = False
	newSC = []
	sig = game.scInGame
	max = len(sig)
	for i,v in enumerate(sig):
		# print i
		window = []
		# print i
		if i-6000 <0:
			start = 0
		else:
			start = i-6000
		if i+6000>=max:
			end = max-1
		else:
			end = i+6000
		window = sig[start:end]
		median = np.median(window)
		newSC.append(v-median)

	plt.plot(newSC)
	plt.plot(sig)
	plt.show()


def plot(game):
	if game.hrv <100:
		# print game.gameid
		# plt.plot(game.firstDiffHR, label="HR First Difference")
		# plt.plot(game.firstDiffSC, label="SC First Difference")
		# plt.plot(game.hrInGame, label="BPM")
		plt.plot(game.scInGame, label="Skin Conductance")
		plt.ylim(-5,10)
		# plt.plot(buff, label='Raw HR Signal')
		# plt.plot(filteredHR, label='Filtered HR Signal')
		# plt.plot(minMovAvg, label='Moving Average')
		# plt.scatter(minPeakList, minYbeat, color="red", label='Detected Peak')
		# # plt.plot(bpms)

		for d in game.dilemmaArray:
			plt.axvline(x=d.initIndex, color="g",  linestyle='--')
			# plt.axvline(x=d.endIndex, color="c", linestyle='--')
			plt.text(d.initIndex + 5, 10, d.dilemmaId, rotation=90, verticalalignment='center')
			# plt.text(d.endIndex + 5, 1, d.dilemmaId, rotation=90, verticalalignment='center')
		plt.legend()
		plt.show()




def run(game):

	# print game.gameid
	rawHR = game.rawHR
	getInGameHR(game)

	for d in game.dilemmaArray:
   		d.setDilemmaIndexes()
	buff = []
	for r in game.rawHRInGame:
		buff.append(math.pow(r,3)) #buffing the signal to make peaks more obvious



	filteredHR = getButterWorth(buff, 1.5 , 2)

	augList = [-20, -10, -5, -1, 1, 5, 10, 15, 20, 25, 30, 40, 50]
	# augList = [1]
	statsDict = {}
	# filteredHR = rawHR
	for k, aug in enumerate(augList):
		mov_avg = getMovAvg(filteredHR, 768, aug) #0.75*1024
		peakList = getPeakDetection(filteredHR, mov_avg)
		ybeat = [filteredHR[x] for x in peakList]
		intervals = getIntervals(peakList)
		rrsd = np.std(intervals)
		sqdiffs = getIntervalSqdiffs(intervals)
		rmssd = getRmssd(sqdiffs)
		bpms = getBpm(intervals, peakList, game)
		
		# game.hrv = rmssd
		temp = {
			'rrsd' : rrsd,
			'bpms' : bpms,
			'hrv' : rmssd,
			'movAvg': mov_avg,
			'peakList' : peakList,
			'ybeat': ybeat
		}
		statsDict[str(k)] = temp
		# print "......"
		# print aug
		# print rrsd
		# print np.mean(bpms)
		# print "......"

	minrrsd = 99999
	minBpms = []
	minMovAvg = []
	minPeakList = []
	minYbeat = []
	for d in statsDict:
		if not math.isnan(np.mean(statsDict[d]['bpms'])) and statsDict[d]['rrsd']<minrrsd and statsDict[d]['rrsd']>0:
			minrrsd = statsDict[d]['rrsd']
			minBpms = statsDict[d]['bpms']
			game.hrv = statsDict[d]['hrv']
			minMovAvg = statsDict[d]['movAvg']
			minPeakList = statsDict[d]['peakList']
			minYbeat = statsDict[d]['ybeat']

	

	getHRStats(game,minBpms)
	getSCStats(game)
	getDilemmaStats(game)



	
	# detoneSC(game)

	# print "..............."
	# print len(game.rawHRInGame)
	# print len(game.hrInGame)
	# print "..............."

	# print "HRV: ", game.hrv
	# print "version:  ", game.version
	# print "avg BPM: ", np.mean(bpms)


	plot(game)



	

	
	
	# print len(filteredHR)
	# print len(bpms)


