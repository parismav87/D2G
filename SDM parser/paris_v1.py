 
import xml.etree.ElementTree as ET
import sys
import os
from datetime import datetime as dt
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt
import pytz
import math 
from hrv.classical import time_domain
from scipy import stats
import hrvcalc

gamesArray = []


def getGameId(root):
	sdmmeta = root.find('sdmmeta')
	gameid = sdmmeta.find('gameid').text.strip()
	return gameid

def getUTCTime(timeString):
	#print(timeString)
	dateTime = dt.strptime(timeString, "%Y-%m-%dT%H:%M:%S")
	# dateTime.replace(tzinfo=pytz.utc)
	# print dateTime.tzinfo
	# dateTime.astimezone(pytz.timezone('Europe/Amsterdam'))
	# # print(dateTime)
	# print dateTime.tzinfo
	utcTime = (dateTime - dt(1970, 1, 1)).total_seconds()
	utcTime = float(utcTime) - 7200 # revert to same timezone
 	# utcTime = dt.timestamp(dateTime)
	#print(utcTime)
	return utcTime

def getSensorStats(game):
	firstMeasurement = False
	lastMeasurement = False
	minHR = 9999
	maxHR = 0
	minSC = 9999
	maxSC = 0
	timestampMaxHR = 0
	timestampMinHR = 0
	timestampMaxSC = 0
	timestampMinSC = 0
	filteredHR = []

	for i,v in enumerate(game.rawHR):
		if game.timestamps[i] >= game.gamestartUTC and game.timestamps[i] <= game.gameendUTC and v>0: #filter out <=0 values
			filteredHR.append(v)

	game.hrv = time_domain(filteredHR)['rmssd']

	for i,v in enumerate(game.hrSignal):
		if game.timestamps[i] >= game.gamestartUTC and game.timestamps[i] <= game.gameendUTC and v>0: #filter out <=0 values
			if not firstMeasurement:
				firstMeasurement = True
				game.initHR = v
			if v> maxHR:
				maxHR = v
				timestampMaxHR = i
			if v< minHR:
				minHR = v
				timestampMinHR = i
			game.rawHRInGame.append(game.rawHR[i])
			game.hrInGame.append(v) # add hr value to ingame HR

		elif game.timestamps[i]> game.gameendUTC and  not lastMeasurement:
			lastMeasurement = True
			game.endHR = game.hrSignal[i-1]

	game.avgHR = np.average(game.hrInGame)
	game.stdHR = np.std(game.hrInGame)

	# plt.plot(game.hrInGame)
 #   	plt.show()

	for h in game.hrInGame:
		game.hrBaseline = h-game.avgHR


	game.maxHR = maxHR - game.avgHR
	game.minHR = minHR - game.avgHR
	game.timestampMaxHR = float(timestampMaxHR)/len(game.timestamps)
	game.timestampMinHR = float(timestampMinHR)/len(game.timestamps)
	game.diffHR = maxHR - minHR
	game.diffTimestampHR = math.fabs(game.timestampMaxHR - game.timestampMinHR)
	


	# print game.maxHR
	# print game.minHR
	# print game.initHR
	# print game.endHR
	# print game.timestampMaxHR
	# print game.timestampMinHR
	# print game.diffHR
	# print game.diffTimestampHR
	# print game.avgHR
	# print game.stdHR
	# print "---------------------"
	#reset for SC
	firstMeasurement = False
	lastMeasurement = False

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
	game.shiftSC = game.endSC - game.initSC
	game.timestampMaxSC = float(timestampMaxSC)/len(game.timestamps)
	game.timestampMinSC = float(timestampMinSC)/len(game.timestamps)
	game.diffSC = maxSC - minSC
	game.diffTimestampSC = math.fabs(game.timestampMaxSC - game.timestampMinSC)
	

	# print game.maxSC
	# print game.minSC
	# print game.initSC
	# print game.endSC
	# print game.shiftSC
	# print game.timestampMaxSC
	# print game.timestampMinSC
	# print game.diffSC
	# print game.diffTimestampSC
	# print game.avgSC
	# print game.stdSC

	# print "====="
	# print game.hrv

class Game(object):
	def __init__(self, gameid):
		self.gamestartUTC = 0
		self.gameendUTC = 0
		self.dilemmaArray = []
		self.empathyArray = []
		self.characterArray = []
		self.infoArray = []
		self.importanceArray = []
		self.adviceArray = []
		self.advicecharArray = [] 
		self.voteadviceArray = []
		self.openDilemmaArray = []
		self.answerDilemmaArray = []
		self.openInfoArray = []
		self.showAdviceArray = []
		self.rootDilemmasArray = []
		self.gameid = gameid
		self.version = -1
		self.sumDilTimeToAnswer = 0
		self.sumDilTimeOpen = 0
		self.sumDilNeutralFeedback = 0
		self.sumDilPositiveFeedback = 0
		self.sumDilNegativeFeedback = 0
		self.sumDilAdviceRequested = 0
		self.sumDilInfosRead = 0
		self.sumDilInfos = 40
		self.sumDilTimesOpened = 0
		self.sumDilDirectInfo = 0
		self.sumDilIndirectInfo = 0
		self.dilTimeToAnswerArr = []
		self.dilTimeOpenArr = []
		self.dilTimesOpenedArr = []
		self.infoTimeOpenArr = []
		self.scSignal = []
		self.hrSignal = []
		self.timestamps = []
		self.scInGame = []
		self.hrInGame = []
		self.rawHR = []
		self.rawHRInGame = []
		self.hrBaseline = []
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


	def getGameVersion(self, root):
		sdmscenario = root.find('sdmscenario')
		title = sdmscenario.find('title').text.strip()
		# print title
		if("0" in title): 
			self.version = 0
		else: 
			self.version = 1
		# print self.version

class OpenDilemma(object):
	def __init__(self, time, gameTime, dilemmaId):
		self.time = time
		self.gameTime = gameTime
		self.dilemmaId = dilemmaId

class AnswerDilemma(object):
	def __init__(self, time, gameTime, dilemmaId):
		self.time = time
		self.gameTime = gameTime
		self.dilemmaId = dilemmaId

class OpenInfo(object):
	def __init__(self, time, gameTime, infoId, dilemmaId):
		self.time = time
		self.gameTime = gameTime
		self.infoId = infoId
		self.dilemmaId = dilemmaId

class ShowAdvice(object):
	def __init__(self, time, gameTime, dilemmaId):
		self.time = time
		self.gameTime = gameTime
		self.dilemmaId = dilemmaId

class RootDilemma(object):
	def __init__(self, displayTime, dilemmaId):
		self.displayTime = displayTime
		self.dilemmaId = dilemmaId

class Dilemma(object):
	def __init__(self, gameId, dilemmaId, timeOpen, answerType, game):
		self.gameId = gameId
		self.dilemmaId = dilemmaId
		self.timeOpen = timeOpen
		self.answerType = answerType
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

	def setTimeToAnswer(self):
		for rd in self.game.rootDilemmasArray:
			if rd.dilemmaId == self.dilemmaId:
				for da in self.game.answerDilemmaArray:
					if da.dilemmaId == self.dilemmaId:
						self.timeToAnswer = int(da.gameTime) - int(rd.displayTime)

	def setTimesOpened(self):
		for od in self.game.openDilemmaArray:
			if od.dilemmaId == self.dilemmaId:
				self.timesOpened += 1

	def setNumberInfosRead(self):
		for ir in self.game.openInfoArray:
			if ir.dilemmaId == self.dilemmaId:
				self.numberInfosRead += 1

	def setAdviceRequested(self):
		for sa in self.game.showAdviceArray: 
			if sa.dilemmaId == self.dilemmaId:
				self.adviceRequested = True

	def setEmpathyFeedback(self):
		for emp in self.game.empathyArray:
			if emp.dilemmaId == self.dilemmaId:
				self.empathyFeedback = emp.mood

	def setDirectInfos(self):
		for imp in self.game.importanceArray:
			if imp.dilemmaId == self.dilemmaId and imp.importanceMoment == "DIRECT":
				self.directInfos += 1

	def setIndirectInfos(self):
		for imp in self.game.importanceArray:
			if imp.dilemmaId == self.dilemmaId and imp.importanceMoment == "INDIRECT":
				self.indirectInfos += 1

	def setNumberAskedInfos(self):
		for c in self.game.characterArray:
			self.numberAskedInfos += int(c.askedInfo)

	def setSumAdvisorMood(self):
		for c in self.game.characterArray:
			self.sumAdvisorMood += int(c.advMood)

	def setSecondaryMeasurements(self):
		self.setTimeToAnswer()
		self.setTimesOpened()
		self.setNumberInfosRead()
		self.setAdviceRequested()
		self.setEmpathyFeedback()
		self.setDirectInfos()
		self.setIndirectInfos()
		self.setNumberAskedInfos()
		self.setSumAdvisorMood()

	def getCSVHeader(self):
		return "gameId, dilemmaId,timeOpen,answerType, timeToAnswer, timesOpened, numberInfosTotal, numberInfosRead, adviceRequested, empathyFeedback, directInfos, indirectInfos"

	def toCSV(self):
		return (self.gameId+","+self.dilemmaId+","+self.timeOpen+","+self.answerType+","+str(self.timeToAnswer)+","+str(self.timesOpened)+","+str(self.numberInfosTotal)
			+","+str(self.numberInfosRead)+","+str(self.adviceRequested)+","+self.empathyFeedback+","+str(self.directInfos)+","+str(self.indirectInfos))
#+","+str(self.numberAskedInfos)+","+str(self.sumAdvisorMood)
class Empathy(object):
	def __init__(self, gameId, dilemmaId, empathyGroup, mood):
		self.gameId = gameId
		self.dilemmaId = dilemmaId
		self.empathyGroup = empathyGroup
		self.mood = mood

	def getCSVHeader(self):
		return "gameId,dilemmaId,empathyGroup,mood"

	def toCSV(self):
		return self.gameId+","+self.dilemmaId+","+self.empathyGroup+","+self.mood

class Character(object):
	def __init__(self, gameId, advisor, askedinfo, advMood):
		self.gameId = gameId
		self.advisor = advisor
		self.askedInfo = askedinfo
		self.advMood = advMood

	def getCSVHeader(self):
		return "gameId,advisor,askedinfo,advMood"

	def toCSV(self):
		return self.gameId+","+self.advisor+","+self.askedInfo+","+self.advMood

class Info(object):
	def __init__(self, dilemmaid, correctanswer,dilemmaempathyscore):
		self.dilemmaid = dilemmaid
		self.correctanswer = correctanswer
		self.dilemmaempathyscore = dilemmaempathyscore

	def getCSVHeader(self):
		return "dilemmaId,correctanswer,dilemmaempathyscore"

	def toCSV(self):
		return self.dilemmaid+","+self.correctanswer+","+self.dilemmaempathyscore

class Importance(object):
	def __init__(self, gameid, dilemmaid, adviceid,importancetype,importancemoment,time):
		self.gameId = gameid
		self.dilemmaId = dilemmaid
		self.adviceId = adviceid
		self.importanceType = importancetype
		self.importanceMoment = importancemoment
		self.time = time
		
	def getCSVHeader(self):
		return "gameId,dilemmaId,adviceid,importancetype,importancemoment,importancetime"

	def toCSV(self):
		return self.gameId+","+self.dilemmaId+","+self.adviceId+","+self.importanceType+","+self.importanceMoment+","+self.time

class Advice(object):
	def __init__(self,gameId,dilemmaId,adviceid,opentime,opengametime):
		self.gameId = gameId
		self.dilemmaId = dilemmaId
		self.adviceid = adviceid
		self.opentime = opentime
		self.opengametime = opengametime

	def getCSVHeader(self):
		return "gameId,dilemmaId,adviceid,opentime,opengametime"

	def toCSV(self):
		return self.gameId+","+self.dilemmaId+","+self.adviceid+","+self.opentime+","+self.opengametime

class AdviceChar(object):
	def __init__(self,adviceid,advisor):
		self.adviceid = adviceid
		self.advisor = advisor

	def getCSVHeader(self):
		return "adviceid,advisor"

	def toCSV(self):
		return self.adviceid+","+self.advisor

class VoteAdvice(object):
	def __init__(self,gameId,dilemmaId,voteadvicetime,vote):
		self.gameId = gameId
		self.dilemmaId = dilemmaId
		self.voteadvicetime = voteadvicetime
		self.vote = vote

	def getCSVHeader(self):
		return "gameId,dilemmaId,voteadvicetime,vote"

	def toCSV(self):
		return self.gameId+","+self.dilemmaId+","+self.voteadvicetime+","+self.vote


def getStats():
	for g in gamesArray:
		if g.version == 0:
			maxHR0.append(g.maxHR)
			minHR0.append(g.minHR)
			avgHR0.append(g.avgHR)
			stdHR0.append(g.stdHR)
			timestampMaxHR0.append(g.timestampMaxHR)
			timestampMinHR0.append(g.timestampMinHR)
			diffTimestampHR0.append(g.diffTimestampHR)
			diffHR0.append(g.diffHR)
			maxSC0.append(g.maxSC)
			minSC0.append(g.minSC)
			avgSC0.append(g.avgSC)
			stdSC0.append(g.stdSC)
			timestampMaxSC0.append(g.timestampMaxSC)
			timestampMinSC0.append(g.timestampMinSC)
			diffTimestampSC0.append(g.diffTimestampSC)
			diffSC0.append(g.diffSC)
			hrv0.append(g.hrv)
		else:
			maxHR1.append(g.maxHR)
			minHR1.append(g.minHR)
			avgHR1.append(g.avgHR)
			stdHR1.append(g.stdHR)
			timestampMaxHR1.append(g.timestampMaxHR)
			timestampMinHR1.append(g.timestampMinHR)
			diffTimestampHR1.append(g.diffTimestampHR)
			diffHR1.append(g.diffHR)
			maxSC1.append(g.maxSC)
			minSC1.append(g.minSC)
			avgSC1.append(g.avgSC)
			stdSC1.append(g.stdSC)
			timestampMaxSC1.append(g.timestampMaxSC)
			timestampMinSC1.append(g.timestampMinSC)
			diffTimestampSC1.append(g.diffTimestampSC)
			diffSC1.append(g.diffSC)
			hrv1.append(g.hrv)


			
	#significance tests
	print "maxHR ---->", stats.ttest_ind(maxHR0, maxHR1)
	print "minHR ---->", stats.ttest_ind(minHR0, minHR1) 
	print "avgHR ---->", stats.ttest_ind(avgHR0, avgHR1) 
	print "stdHR ---->", stats.ttest_ind(stdHR0, stdHR1) 
	print "timestampMaxHR ---->", stats.ttest_ind(timestampMaxHR0, timestampMaxHR1) 
	print "timestampMinHR ---->", stats.ttest_ind(timestampMinHR0, timestampMinHR1) 
	print "diffTimestampHR ---->", stats.ttest_ind(diffTimestampHR0, diffTimestampHR1) 
	print "diffHR1 ---->", stats.ttest_ind(diffHR0, diffHR1) 
	print "maxSC ---->", stats.ttest_ind(maxSC0, maxSC1) 
	print "minSC ---->", stats.ttest_ind(minSC0, minSC1) 
	print "avgSC ---->", stats.ttest_ind(avgSC0, avgSC1) 
	print "stdSC ---->", stats.ttest_ind(stdSC0, stdSC1) 
	print "timestampMaxSC ---->", stats.ttest_ind(timestampMaxSC0, timestampMaxSC1) 
	print "timestampMinSC ---->", stats.ttest_ind(timestampMinSC0, timestampMinSC1) 
	print "diffTimestampSC ---->", stats.ttest_ind(diffTimestampSC0, diffTimestampSC1) 
	print "diffSC ---->", stats.ttest_ind(diffSC0, diffSC1) 
	print "hrv ---->", stats.ttest_ind(hrv0, hrv1) 

	avgMaxHR0 = np.average(maxHR0)
	avgMinHR0 = np.average(minHR0)
	avgAvgHR0 = np.average(avgHR0)
	avgStdHR0 = np.average(stdHR0)
	avgTimestampMaxHR0 = np.average(timestampMaxHR0)
	avgTimestampMinHR0 = np.average(timestampMinHR0)
	avgDiffTimestampHR0 = np.average(diffTimestampHR0)
	avgDiffHR0 = np.average(diffHR0)
	avgMaxSC0 = np.average(maxSC0)
	avgMinSC0 = np.average(minSC0)
	avgAvgSC0 = np.average(avgSC0)
	avgStdSC0 = np.average(stdSC0)
	avgTimestampMaxSC0 = np.average(timestampMaxSC0)
	avgTimestampMinSC0 = np.average(timestampMinSC0)
	avgDiffTimestampSC0 = np.average(diffTimestampSC0)
	avgDiffSC0 = np.average(diffSC0)
	avgHrv0 = np.average(hrv0)

	avgMaxHR1 = np.average(maxHR1)
	avgMinHR1 = np.average(minHR1)
	avgAvgHR1 = np.average(avgHR1)
	avgStdHR1 = np.average(stdHR1)
	avgTimestampMaxHR1 = np.average(timestampMaxHR1)
	avgTimestampMinHR1 = np.average(timestampMinHR1)
	avgDiffTimestampHR1 = np.average(diffTimestampHR1)
	avgDiffHR1 = np.average(diffHR1)
	avgMaxSC1 = np.average(maxSC1)
	avgMinSC1 = np.average(minSC1)
	avgAvgSC1 = np.average(avgSC1)
	avgStdSC1 = np.average(stdSC1)
	avgTimestampMaxSC1 = np.average(timestampMaxSC1)
	avgTimestampMinSC1 = np.average(timestampMinSC1)
	avgDiffTimestampSC1 = np.average(diffTimestampSC1)
	avgDiffSC1 = np.average(diffSC1)
	avgHrv1 = np.average(hrv1)


	print avgMaxHR0, " -- ", avgMaxHR1, " avg max hr"
	print avgMinHR0, " -- ", avgMinHR1, " avg min hr"
	print avgAvgHR0, " -- ", avgAvgHR1, " avg avg hr"
	print avgStdHR0, " -- ", avgStdHR1, " avg std hr"
	print avgTimestampMaxHR0, " -- ", avgTimestampMaxHR1, " avg timestamp max hr"
	print avgTimestampMinHR0, " -- ", avgTimestampMinHR1, " avg timestamp min hr"
	print avgDiffTimestampHR0, " -- ", avgDiffTimestampHR1, " avg diff timestamp hr"
	print avgDiffHR0, " -- ", avgDiffHR1, " avg diff hr"
	print avgHrv0, " -- ", avgHrv1, " avg hrv"
	print avgMaxSC0, " -- ", avgMaxSC1, " avg max sc"
	print avgMinSC0, " -- ", avgMinSC1, " avg min sc"
	print avgAvgSC0, " -- ", avgAvgSC1, " avg avg sc"
	print avgStdSC0, " -- ", avgStdSC1, " avg std sc"
	print avgTimestampMaxSC0, " -- ", avgTimestampMaxSC1, " avg timestamp max sc"
	print avgTimestampMinSC0, " -- ", avgTimestampMinSC1, " avg timastamp min sc"
	print avgDiffTimestampSC0, " -- ", avgDiffTimestampSC1, " avg diff timestamp sc"
	print avgDiffSC0, " -- ", avgDiffSC1, " avg diff sc"



def parseGameTime(game):
	for sdmmeta in root.iter('sdmmeta'):
		gamestart = sdmmeta.find('gamestart').text.strip()
		gamestartUTC = getUTCTime(gamestart)
		game.gamestartUTC = gamestartUTC

		gameend = sdmmeta.find('gameend').text.strip()
		gameendUTC = getUTCTime(gameend)
		game.gameendUTC = gameendUTC

def parseDilemmas(game):
	for sdmfeedback in root.iter('sdmfeedback'):
		for dilemmas in sdmfeedback.iter('dilemmas'):
			for gamedilemma in dilemmas.iter('gamedilemma'):

				gameId = getGameId(root)
				dilemmaId = gamedilemma.find('dilemmatitle').text.strip()
				timeOpen = gamedilemma.attrib.get('timeOpen')
				answerType = gamedilemma.attrib.get('answerType')

				d = Dilemma(gameId, dilemmaId, timeOpen, answerType, game)
				game.dilemmaArray.append(d)

def parseRootDilemmas(game):
	for sc in root.iter('sdmscenario'):
		for sl in sc.iter('storyline'):
			for dil in sl.iter('dilemma'):
				displayTime = dil.attrib.get('displaytime')
				dilemmaId = dil.find('name').text.strip()

				d = RootDilemma(displayTime, dilemmaId)
				game.rootDilemmasArray.append(d)

def parseOpenDilemmas(game): #time gametime dilemmaid
	for sdmuseraction in root.iter('sdmuseraction'):
		for od in sdmuseraction.iter('openDilemma'):
			time = od.find('time').text.strip()
			gameTime = od.find('gameTime').text.strip()
			for dil in od.iter('dilemma'):
				dilemmaId = dil.find('name').text.strip()
			d = OpenDilemma(time, gameTime, dilemmaId)
			game.openDilemmaArray.append(d)

def parseShowAdvice(game): #time gametime dilemmaid
	for sdmuseraction in root.iter('sdmuseraction'):
		for od in sdmuseraction.iter('showAdvice'):
			time = od.find('time').text.strip()
			gameTime = od.find('gameTime').text.strip()
			for dil in od.iter('dilemma'):
				dilemmaId = dil.find('name').text.strip()
			d = ShowAdvice(time, gameTime, dilemmaId)
			game.showAdviceArray.append(d)

def parseAnswerDilemmas(game): 
	for sdmuseraction in root.iter('sdmuseraction'):
		for od in sdmuseraction.iter('answerDilemma'):
			time = od.find('time').text.strip()
			gameTime = od.find('gameTime').text.strip()
			for dil in od.iter('dilemma'):
				dilemmaId = dil.find('name').text.strip()
			d = AnswerDilemma(time, gameTime, dilemmaId)
			game.answerDilemmaArray.append(d)

def parseOpenInfos(game): 
	for sdmuseraction in root.iter('sdmuseraction'):
		for oi in sdmuseraction.iter('openInfo'):
			time = oi.find('time').text.strip()
			gameTime = oi.find('gameTime').text.strip()
			for info in oi.iter('info'):
				infoId = info.find('name').text.strip()
			for il in oi.iter('infoLocation'):
				for dil in il.iter('dilemma'):
					dilemmaId = dil.find('name').text.strip()

			d = OpenInfo(time, gameTime, infoId, dilemmaId)
			game.openInfoArray.append(d)

def parseEmpathy(game):
	for sdmfeedback in root.iter('sdmfeedback'):
		for empathies in sdmfeedback.iter('empathies'):
			for dilemma in empathies.iter('gamedilemma'):
				for emgroups in dilemma.iter('empathygroup'):

					gameId = getGameId(root)
					dilemmaId = dilemma.find('dilemmatitle').text.strip()
					empathyGroup = emgroups.find('name').text.strip()
					mood = emgroups.find('mood').text.strip()

					e = Empathy(gameId, dilemmaId, empathyGroup, mood)
					game.empathyArray.append(e)

def parseCharacter(game):
	for sdmfeedback in root.iter('sdmfeedback'):
		for sources in sdmfeedback.iter('sources'):
			for source in sources.iter('source'):
				
				gameId = getGameId(root)
				advisor = source.find('name').text.strip()
				askedinfo = source.attrib.get('askedinformation')
				advMood = source.attrib.get('mood')

				c = Character(gameId,advisor,askedinfo,advMood)
				game.characterArray.append(c)

def parseInfo(game):
	for sdmscenario in root.iter('sdmscenario'):
		for dilemma in sdmscenario.iter('dilemma'):
			for dilem in dilemma.iter('dilemmaempathyscores'):
				for dilscore in dilem.iter('dilemmaempathyscore'):

					dilemmaid = dilemma.find('name').text.strip()
					correctanswer = dilemma.find('correctanswer').text.strip()
					dilemmaempathyscore = dilscore.attrib.get('score')
				
					i = Info(dilemmaid,correctanswer,dilemmaempathyscore)
					game.infoArray.append(i)

def parseImportance(game):
	for sdmuseraction in root.iter('sdmuseraction'):
		for importance in sdmuseraction.iter('markInfoImportance'):
			for dilem in importance.iter('dilemma'):
				for info in importance.iter('info'):

					gameid = getGameId(root)
					dilemmaid = dilem.find('name').text.strip()
					adviceid = info.find('name').text.strip()
					importancetype = importance.find('importanceType').text.strip()
					importancemoment = importance.find('importanceMoment').text.strip()
					time = importance.find('time').text.strip()
				
					imp = Importance(gameid,dilemmaid,adviceid,importancetype,importancemoment,time)
					game.importanceArray.append(imp)

def parseAdvice(game):
	for sdmuseraction in root.iter('sdmuseraction'):
		for openinfo in sdmuseraction.iter('openInfo'):
			for dilem in openinfo.iter('dilemma'):
				for info in openinfo.iter('info'):

					gameid = getGameId(root)
					dilemmaid = dilem.find('name').text.strip()
					adviceid = info.find('name').text.strip()
					opentime = openinfo.find('time').text.strip()
					opengametime = openinfo.find('gameTime').text.strip()

					a = Advice(gameid,dilemmaid,adviceid,opentime,opengametime)
					game.adviceArray.append(a)

def parseAdviceChar(game):
	for sdmscenario in root.iter('sdmscenario'):
		for dilemma in sdmscenario.iter('dilemma'):
			for infos in dilemma.iter('infos'):
				for info in infos.iter('info'):

					adviceid = info.find('name').text.strip()
					advisor = info.find('virtualcharactername').text.strip()
					advisor = advisor.replace('\n', '')

					ac = AdviceChar(adviceid,advisor)
					game.advicecharArray.append(ac)

def parseVoteAdvice(game):
	for sdmuseraction in root.iter('sdmuseraction'):
		for showadvice in sdmuseraction.iter('showAdvice'):
			for dilem in showadvice.iter('dilemma'):

				gameid = getGameId(root)
				dilemmaid = dilem.find('name').text.strip()
				voteadvicetime = showadvice.find('time').text.strip()
				vote = 'yes'

				v = VoteAdvice(gameid,dilemmaid,voteadvicetime,vote)
				game.voteadviceArray.append(v)

def createCSV(game):

	sdmmeta = root.find('sdmmeta')
	gameid = sdmmeta.find('gameid').text.strip()
	foldername = "output/experimentout"+gameid
	if not os.path.exists(foldername): 
		os.makedirs(foldername)

	dilemmaFile = open(foldername+"/dilemmas.csv", "w+")
	dilemmaFile.write(dilemmaArray[0].getCSVHeader()+"\n")
	for dilemma in dilemmaArray:
		dilemma.setSecondaryMeasurements()
		dilemmaFile.write(dilemma.toCSV().strip()+"\n")

	empathyFile = open(foldername+"/empathy.csv", "w+")
	empathyFile.write(empathyArray[0].getCSVHeader()+"\n")
	for empathy in empathyArray:
		empathyFile.write(empathy.toCSV().strip()+"\n")

	characterFile = open(foldername+"/character.csv", "w+")
	characterFile.write(characterArray[0].getCSVHeader()+"\n")
	for character in characterArray:
		characterFile.write(character.toCSV().strip()+"\n")

	infoFile = open(foldername+"/info.csv", "w+")
	infoFile.write(infoArray[0].getCSVHeader()+"\n")
	for info in infoArray:
		infoFile.write(info.toCSV().strip()+"\n")

	importanceFile = open(foldername+"/importance.csv", "w+")
	importanceFile.write(importanceArray[0].getCSVHeader()+"\n")
	for importance in importanceArray:
		importanceFile.write(importance.toCSV().strip()+"\n")

	adviceFile = open(foldername+"/advice.csv", "w+")
	adviceFile.write(adviceArray[0].getCSVHeader()+"\n")
	for advice in adviceArray:
		adviceFile.write(advice.toCSV().strip()+"\n")

	advicecharFile = open(foldername+"/advicechar.csv", "w+")
	advicecharFile.write(advicecharArray[0].getCSVHeader()+"\n")
	for advicechar in advicecharArray:
		advicecharFile.write(advicechar.toCSV().strip()+"\n")

	voteadviceFile = open(foldername+"/voteadvice.csv","w+")
	voteadviceFile.write(voteadviceArray[0].getCSVHeader()+"\n")
	for voteadvice in voteadviceArray:
		voteadviceFile.write(voteadvice.toCSV().strip()+"\n")


def parseXML(game):
	parseDilemmas(game)
	parseEmpathy(game)
	parseCharacter(game)
	parseInfo(game)
	parseImportance(game)
	parseAdvice(game)
	parseAdviceChar(game)
	parseVoteAdvice(game)
	parseOpenDilemmas(game)
	parseAnswerDilemmas(game)
	parseOpenInfos(game)
	parseShowAdvice(game)
	parseRootDilemmas(game)
	parseGameTime(game)


def calculateStats(game):

	for dil in game.dilemmaArray:

		dil.setSecondaryMeasurements()

		game.dilTimeToAnswerArr.append(int(dil.timeToAnswer))
		game.dilTimeOpenArr.append(int(dil.timeOpen))
		game.dilTimesOpenedArr.append(int(dil.timesOpened))

		game.avgTimeToAnswer = np.average(game.dilTimeToAnswerArr)
		game.avgDilTimeOpen = np.average(game.dilTimeOpenArr)
		game.avgDilTimesOpened = np.average(game.dilTimesOpenedArr)

		game.stdTimeToAnswer = np.std(game.dilTimeToAnswerArr)
		game.stdDilTimeOpen = np.std(game.dilTimeOpenArr)
		game.stdDilTimesOpened = np.std(game.dilTimesOpenedArr)

		game.sumDilInfosRead += int(dil.numberInfosRead)
		game.sumDilDirectInfo += int(dil.directInfos)
		game.sumDilIndirectInfo += int(dil.indirectInfos)

		if dil.empathyFeedback == "NEUTRAL":
			game.sumDilNeutralFeedback += 1
		elif dil.empathyFeedback == "POSITIVE":
			game.sumDilPositiveFeedback += 1
		else: 
			game.sumDilNegativeFeedback += 1

		if dil.adviceRequested:
			game.sumDilAdviceRequested += 1

		

	game.openInfoArray.sort(key=lambda x: int(x.gameTime)) #sort by gametime
	temp = 0
	for i in range(0,len(game.openInfoArray)):
		if i == len(game.openInfoArray)-1:
			for d in game.answerDilemmaArray:
				if d.dilemmaId == game.openInfoArray[i].dilemmaId:
					game.infoTimeOpenArr.append(int(d.gameTime) - int(game.openInfoArray[i].gameTime) + temp)
		else:
			if game.openInfoArray[i+1].dilemmaId == game.openInfoArray[i].dilemmaId: #same dilemma id
				temp += int(game.openInfoArray[i+1].gameTime) - int(game.openInfoArray[i].gameTime)
			else: #find dilemma answer time 
				for d in game.answerDilemmaArray:
					if d.dilemmaId == game.openInfoArray[i].dilemmaId:
						game.infoTimeOpenArr.append(int(d.gameTime) - int(game.openInfoArray[i].gameTime)+temp)
						temp = 0
	game.avgInfoOpen = np.average(game.infoTimeOpenArr)
	game.stdInfoOpen = np.std(game.infoTimeOpenArr)

	sdmmeta = root.find('sdmmeta')
	gameid = sdmmeta.find('gameid').text.strip()
	foldername = "output/experimentout"+gameid

	if not os.path.exists(foldername): 
		os.makedirs(foldername)
	statsFile = open(foldername+"/stats.csv", "w+")
	statsFile.write("sumAdvisorMood,averageInfoOpen,stdInfoOpen,averageTimeToAnswer,stdTimeToAnswer,averageTimeDilemmaOpen,stdTimeDilemmaOpen,sumPosFeedback,sumNegFeedback,sumNeuFeedback,sumAdviceRequested,sumInfosRead,sumInfos,avgTimesDilemmaOpened,stdTimesDilemmaOpened,sumDirectInfo,sumIndirectInfo"+"\n")
	statsFile.write(str(game.dilemmaArray[0].sumAdvisorMood)+","+str(game.avgInfoOpen)+","+str(game.stdInfoOpen)+","+str(game.avgTimeToAnswer)+","+str(game.stdTimeToAnswer)+","+str(game.avgDilTimeOpen)+","+str(game.stdDilTimeOpen)
		+","+str(game.sumDilPositiveFeedback)+","+str(game.sumDilNegativeFeedback)+","+str(game.sumDilNeutralFeedback)+","+str(game.sumDilAdviceRequested)+","+str(game.sumDilInfosRead)
		+","+str(game.sumDilInfos)+","+str(game.avgDilTimesOpened)+","+str(game.stdDilTimesOpened)+","+str(game.sumDilDirectInfo)+","+str(game.sumDilIndirectInfo))




#-------   MAIN   -----------------	
if len(sys.argv)>1: 
	filePath = sys.argv[1]
	games = ET.parse(filePath).getroot().iter('sdmgame')

	for root in games : 		
		gameid = getGameId(root)
		g = Game(gameid)
		gamesArray.append(g)
		g.getGameVersion(root)
		parseXML(g)
		# createCSV(g)
		calculateStats(g)
		# print g.sumDilInfosRead

	d1 = "Sensor-Data"
 	for f1 in os.listdir(d1):
 		# print f1
 		d2 = os.path.join(d1,f1)
 		for f2 in os.listdir(d2):
 			# print f2
 			d3 = os.path.join(d2,f2)
 			for f3 in os.listdir(d3):
 				# print f3
 				if ".csv" in f3:
 					arr = f3.split("_")
 					gameid = arr[len(arr)-1].split(".")[0] #exclude ".csv"
 					# print gameid
 					filepath = os.path.join(d3,f3)

 					found = False
 					for game in gamesArray:
 					# game = gamesArray[2]
						if gameid == game.gameid:
							found = True
							# print game.gamestartUTC
							# print game.gameendUTC
							# print "======================"
							# print "found game " + gameid
							with open(filepath, 'rb') as csvfile:
						   		reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
						   		i = 0
						   		sc = []
						   		hr = []
						   		timestamps = []
						   		rawHR = []
						   		for row in reader:
						   			if i>2:
						   				rawHR.append(float(row[4]))
						   				sc.append(round(float(row[2]),2))
						   				hr.append(round(float(row[6]),2))
						   				timestamps.append(float(row[0])/1000) # no millis
						   				# print row[0]
						   				i+=1
						   			else: 
						   				i+=1
						   	# plt.plot(sc)
						   	# plt.axis([0, i, 0, 20])
						   	# plt.show()
						   	game.scSignal = sc
						   	game.hrSignal = hr
						   	game.timestamps = timestamps
						   	game.rawHR = rawHR
						   	# getSensorStats(game)
						   	# print game.version
						   	hrvcalc.run(game)
						   	# print "++++++++++++++++++++"
						   	# plt.plot(bpms)
						   	# plt.plot(game.hrInGame)
						   	# plt.show()
							   	# print game.avgHR, ",", game.version
						   	# print game.maxHR, " - ",  game.minHR ," - ", game.avgHR ," - ", game.stdHR ," - ", game.timestampMaxHR ," - ", game.timestampMinHR ," - ", game.diffTimestampHR ," - ", game.diffHR ," - ", game.hrv ," - ", game.version
						   	# print game.maxHR, " - ",  game.minSC ," - ", game.avgSC ," - ", game.stdSC ," - ", game.timestampMaxSC ," - ", game.timestampMinSC ," - ", game.diffTimestampSC ," - ", game.diffSC ," - ", game.version
					if not found:
						print "game not found ", gameid
	maxHR0, minHR0, avgHR0, stdHR0, timestampMaxHR0, timestampMinHR0, diffTimestampHR0, diffHR0, maxSC0, minSC0, avgSC0, stdSC0, timestampMaxSC0, timestampMinSC0, diffTimestampSC0, diffSC0, hrv0 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
	maxHR1, minHR1, avgHR1, stdHR1, timestampMaxHR1, timestampMinHR1, diffTimestampHR1, diffHR1, maxSC1, minSC1, avgSC1, stdSC1, timestampMaxSC1, timestampMinSC1, diffTimestampSC1, diffSC1, hrv1 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
	



	for k,v in enumerate(gamesArray): #clean up false data (user not reading anything/no sensor data)
		print ' :::::::::::::::::::::   ', v.avgHR
		print ' :::::::::::::::::::::   ', v.sumDilInfosRead
		if v.avgHR == 0  or v.sumDilInfosRead <2 : #0 or 1 infos read lol
			gamesArray.pop(k)
			print "popped game ", v.gameid


	print "number of games: ", len(gamesArray)
	getStats()




	# for game in gamesArray: 
	# 	if game.sensorData == []:
	# 		print game.gameid

	# print gamesArray[0].timestamps[0]
	# print gamesArray[0].gamestartUTC
	# print gamesArray[0].timestamps[0] - gamesArray[0].gamestartUTC


	# newsig = []
	# # movavg = []
	# for k,v in enumerate(gamesArray[0].hrSignal):
	# 	if v>0:
	# 		newsig.append(v)
	# # print gamesArray[0].sensorData[1]


	# plt.plot(newsig)
	# plt.plot(gamesArray[0].scSignal)
	# # plt.plot(movavg)	
	# plt.show()
	
			
	print("Finished parsing.")
else:
	print("Please enter file path. Example: >> parser.py MYFILE.xml")