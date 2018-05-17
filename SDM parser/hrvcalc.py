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
			augmented = m#to avoid peak detection in 2nd heart contraction
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
			intervals.append(interval)			
	intervals.append(peakList[len(peakList)-1] - peakList[len(peakList)-2]) #last element
	return intervals


def getBpm(intervals, peakList):
	bpms = []
	start = 0
	for k,v in enumerate(intervals):
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
	nyq = 0.5 * 1000 #Nyquist frequeny is half the sampling frequency
	normal_cutoff = cutoff / nyq
	b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
	y = signal.lfilter(b, a, data)
	return y


def run(rawHR):
	buff = []
	for r in rawHR:
		buff.append(math.pow(r,3))

	filteredHR = getButterWorth(buff, 1.5 , 3)
	# filteredHR = rawHR
	mov_avg = getMovAvg(filteredHR, 750)
	peakList = getPeakDetection(filteredHR, mov_avg)
	ybeat = [filteredHR[x] for x in peakList]
	intervals = getIntervals(peakList)
	sqdiffs = getIntervalSqdiffs(intervals)
	rmssd = getRmssd(sqdiffs)
	bpms = getBpm(intervals, peakList)

	print "HRV: ", rmssd
	print "avg BPM: ", np.mean(bpms)
	# plt.ion()
	# plt.plot(rawHR)
	# plt.plot(filteredHR)
	# plt.plot(mov_avg)
	# plt.scatter(peakList, ybeat, color="red")
	# plt.plot(bpms)
	# plt.show()

	return bpms
	# print len(filteredHR)
	# print len(bpms)