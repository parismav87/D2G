import csv

import matplotlib.pyplot as plt
import numpy as np
import math


def movAvg(dataList, window):
	movAvg = []
	cumSum = [0]
	avg = np.mean(dataList)
	for k,v in enumerate(dataList,1):
		cumSum.append(cumSum[k-1] + v) 
		if k >= window:
			m = (cumSum[k] - cumSum[k-window]) / window
			augmented = m*1.1 #to avoid peak detection in 2nd heart contraction
			movAvg.append(augmented)
		else:
			movAvg.append(avg)
	return movAvg

def peakDetection(dataList, movAvg):
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


def intervals(peakList):
	intervals = []
	for k,v in enumerate(peakList):
		if k < len(peakList)-1:
			interval = peakList[k+1] - v  #interval is already in ms, sampling freq is 1000
			intervals.append(interval)			
	intervals.append(peakList[len(peakList)-1] - peakList[len(peakList)-2]) #last element
	return intervals


def bpm(intervals):
	count = 0
	bpms = []
	for k,v in enumerate(intervals):
		bpms.append(60000/v) #transform to beats per sec
	return bpms

def intervalDiffs(intervals):
	intervalDiffs = []
	for k,v in enumerate(intervals):
		if k>0:
			intervalDiffs.append(abs(intervals[k] - intervals[k-1]))
	return intervalDiffs

def intervalSqdiffs(intervals):
	intervalSqdiffs = []
	for k,v in enumerate(intervals):
		if k>0:
			intervalSqdiffs.append(math.pow(intervals[k] - intervals[k-1], 2))
	return intervalSqdiffs

def rmssd(intervalSqdiffs):
	return np.sqrt(np.mean(intervalSqdiffs))

with open("example.csv", 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
	i = 0
	rawHR = []
	for row in reader:
		if i>2:
			rawHR.append(float(row[4]))
			i+=1
		else: 
			i+=1

mov_avg = movAvg(rawHR, 750)
peakList = peakDetection(rawHR, mov_avg)
ybeat = [rawHR[x] for x in peakList]
intervals = intervals(peakList)
sqdiffs = intervalSqdiffs(intervals)
rmssd = rmssd(sqdiffs)
bpms = bpm(intervals)

# print sqdiffs
print rmssd
plt.plot(rawHR)
plt.plot(mov_avg)
plt.scatter(peakList, ybeat, color="red")
# plt.plot(bpms)
plt.show()