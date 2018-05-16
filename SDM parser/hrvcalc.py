import csv

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.signal import butter, lfilter

def getMovAvg(dataList, window):
	movAvg = []
	avg = np.mean(dataList)
	cumSum = [0]
	
	for k,v in enumerate(dataList,1):
		cumSum.append(cumSum[k-1] + v) 
		if k >= window:
			m = (cumSum[k] - cumSum[k-window]) / window
			augmented = m*1.1 #to avoid peak detection in 2nd heart contraction
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
	return peakList


def getIntervals(peakList):
	intervals = []
	for k,v in enumerate(peakList):
		if k < len(peakList)-1:
			interval = peakList[k+1] - v  #interval is already in ms, sampling freq is 1000
			intervals.append(interval)			
	intervals.append(peakList[len(peakList)-1] - peakList[len(peakList)-2]) #last element
	return intervals


def getBpm(intervals):
	count = 0
	bpms = []
	for k,v in enumerate(intervals):
		bpms.append(60000/v) #transform to beats per sec
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
	nyq = 0.5 * 1000 #Nyquist frequeny is half the sampling frequency
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = lfilter(b, a, data)
	return y


def run(rawHR):
	
	buff = []
	for r in rawHR:
		buff.append(math.pow(r,3))
	filteredHR = getButterWorth(buff, 2.5, 5)
	mov_avg = getMovAvg(filteredHR, 750)
	peakList = getPeakDetection(filteredHR, mov_avg)
	ybeat = [filteredHR[x] for x in peakList]
	intervals = getIntervals(peakList)
	sqdiffs = getIntervalSqdiffs(intervals)
	rmssd = getRmssd(sqdiffs)
	bpms = getBpm(intervals)




	# print sqdiffs
	print rmssd
	# plt.plot(rawHR)
	plt.plot(filteredHR)
	plt.plot(mov_avg)
	plt.scatter(peakList, ybeat, color="red")
	# plt.plot(bpms)
	plt.show()