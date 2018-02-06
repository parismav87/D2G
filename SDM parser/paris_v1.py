 
import xml.etree.ElementTree as ET
import sys
import os
from datetime import datetime as dt

dilemmaArray    = []
empathyArray    = []
characterArray  = []
infoArray       = []
importanceArray = []
adviceArray     = []
advicecharArray = []
voteadviceArray = []
openDilemmaArray = []
answerDilemmaArray = []
openInfoArray = []
showAdviceArray = []
rootDilemmasArray = []

def getGameId(root):
	sdmgame	= root.find('sdmgame')
	sdmmeta = sdmgame.find('sdmmeta')
	gameid = sdmmeta.find('gameid').text.strip()
	return gameid

def getUTCTime(timeString):
	#print(timeString)
	dateTime = dt.strptime(timeString, "%Y-%m-%dT%H:%M:%S.%f")
	#print(dateTime)
	utcTime = dt.timestamp(dateTime)
	#print(utcTime)
	return utcTime

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
	def __init__(self, gameId, dilemmaId, timeOpen, answerType):
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

	def setTimeToAnswer(self):
		for rd in rootDilemmasArray:
			if rd.dilemmaId == self.dilemmaId:
				for da in answerDilemmaArray:
					if da.dilemmaId == self.dilemmaId:
						self.timeToAnswer = int(da.gameTime) - int(rd.displayTime)

	def setTimesOpened(self):
		for od in openDilemmaArray:
			if od.dilemmaId == self.dilemmaId:
				self.timesOpened += 1

	def setNumberInfosRead(self):
		for ir in openInfoArray:
			if ir.dilemmaId == self.dilemmaId:
				self.numberInfosRead += 1

	def setAdviceRequested(self):
		for sa in showAdviceArray: 
			if sa.dilemmaId == self.dilemmaId:
				self.adviceRequested = True

	def setEmpathyFeedback(self):
		for emp in empathyArray:
			if emp.dilemmaId == self.dilemmaId:
				self.empathyFeedback = emp.mood

	def setDirectInfos(self):
		for imp in importanceArray:
			if imp.dilemmaId == self.dilemmaId and imp.importanceMoment == "DIRECT":
				self.directInfos += 1

	def setIndirectInfos(self):
		for imp in importanceArray:
			if imp.dilemmaId == self.dilemmaId and imp.importanceMoment == "INDIRECT":
				self.indirectInfos += 1

	def setNumberAskedInfos(self):
		for c in characterArray:
			self.numberAskedInfos += int(c.askedInfo)

	def setSumAdvisorMood(self):
		for c in characterArray:
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

def parseDilemmas():
	for sdmfeedback in root.iter('sdmfeedback'):
		for dilemmas in sdmfeedback.iter('dilemmas'):
			for gamedilemma in dilemmas.iter('gamedilemma'):

				gameId = getGameId(root)
				dilemmaId = gamedilemma.find('dilemmatitle').text.strip()
				timeOpen = gamedilemma.attrib.get('timeOpen')
				answerType = gamedilemma.attrib.get('answerType')

				d = Dilemma(gameId, dilemmaId, timeOpen, answerType)
				dilemmaArray.append(d)

def parseRootDilemmas():
	for sc in root.iter('sdmscenario'):
		for sl in sc.iter('storyline'):
			for dil in sl.iter('dilemma'):
				displayTime = dil.attrib.get('displaytime')
				dilemmaId = dil.find('name').text.strip()

				d = RootDilemma(displayTime, dilemmaId)
				rootDilemmasArray.append(d)

def parseOpenDilemmas(): #time gametime dilemmaid
	for sdmuseraction in root.iter('sdmuseraction'):
		for od in sdmuseraction.iter('openDilemma'):
			time = od.find('time').text.strip()
			gameTime = od.find('gameTime').text.strip()
			for dil in od.iter('dilemma'):
				dilemmaId = dil.find('name').text.strip()
			d = OpenDilemma(time, gameTime, dilemmaId)
			openDilemmaArray.append(d)

def parseShowAdvice(): #time gametime dilemmaid
	for sdmuseraction in root.iter('sdmuseraction'):
		for od in sdmuseraction.iter('showAdvice'):
			time = od.find('time').text.strip()
			gameTime = od.find('gameTime').text.strip()
			for dil in od.iter('dilemma'):
				dilemmaId = dil.find('name').text.strip()
			d = ShowAdvice(time, gameTime, dilemmaId)
			showAdviceArray.append(d)

def parseAnswerDilemmas(): 
	for sdmuseraction in root.iter('sdmuseraction'):
		for od in sdmuseraction.iter('answerDilemma'):
			time = od.find('time').text.strip()
			gameTime = od.find('gameTime').text.strip()
			for dil in od.iter('dilemma'):
				dilemmaId = dil.find('name').text.strip()
			d = AnswerDilemma(time, gameTime, dilemmaId)
			answerDilemmaArray.append(d)

def parseOpenInfos(): 
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
			openInfoArray.append(d)

def parseEmpathy():
	for sdmfeedback in root.iter('sdmfeedback'):
		for empathies in sdmfeedback.iter('empathies'):
			for dilemma in empathies.iter('gamedilemma'):
				for emgroups in dilemma.iter('empathygroup'):

					gameId = getGameId(root)
					dilemmaId = dilemma.find('dilemmatitle').text.strip()
					empathyGroup = emgroups.find('name').text.strip()
					mood = emgroups.find('mood').text.strip()

					e = Empathy(gameId, dilemmaId, empathyGroup, mood)
					empathyArray.append(e)

def parseCharacter():
	for sdmfeedback in root.iter('sdmfeedback'):
		for sources in sdmfeedback.iter('sources'):
			for source in sources.iter('source'):
				
				gameId = getGameId(root)
				advisor = source.find('name').text.strip()
				askedinfo = source.attrib.get('askedinformation')
				advMood = source.attrib.get('mood')

				c = Character(gameId,advisor,askedinfo,advMood)
				characterArray.append(c)

def parseInfo():
	for sdmscenario in root.iter('sdmscenario'):
		for dilemma in sdmscenario.iter('dilemma'):
			for dilem in dilemma.iter('dilemmaempathyscores'):
				for dilscore in dilem.iter('dilemmaempathyscore'):

					dilemmaid = dilemma.find('name').text.strip()
					correctanswer = dilemma.find('correctanswer').text.strip()
					dilemmaempathyscore = dilscore.attrib.get('score')
				
					i = Info(dilemmaid,correctanswer,dilemmaempathyscore)
					infoArray.append(i)

def parseImportance():
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
					importanceArray.append(imp)

def parseAdvice():
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
					adviceArray.append(a)

def parseAdviceChar():
	for sdmscenario in root.iter('sdmscenario'):
		for dilemma in sdmscenario.iter('dilemma'):
			for infos in dilemma.iter('infos'):
				for info in infos.iter('info'):

					adviceid = info.find('name').text.strip()
					advisor = info.find('virtualcharactername').text.strip()
					advisor = advisor.replace('\n', '')

					ac = AdviceChar(adviceid,advisor)
					advicecharArray.append(ac)

def parseVoteAdvice():
	for sdmuseraction in root.iter('sdmuseraction'):
		for showadvice in sdmuseraction.iter('showAdvice'):
			for dilem in showadvice.iter('dilemma'):

				gameid = getGameId(root)
				dilemmaid = dilem.find('name').text.strip()
				voteadvicetime = showadvice.find('time').text.strip()
				vote = 'yes'

				v = VoteAdvice(gameid,dilemmaid,voteadvicetime,vote)
				voteadviceArray.append(v)

def createCSV():
	if not os.path.exists("experimentout"): 
		os.makedirs("experimentout")

	dilemmaFile = open("experimentout/dilemmas.csv", "w+")
	dilemmaFile.write(dilemmaArray[0].getCSVHeader()+"\n")
	for dilemma in dilemmaArray:
		dilemma.setSecondaryMeasurements()
		dilemmaFile.write(dilemma.toCSV().strip()+"\n")

	empathyFile = open("experimentout/empathy.csv", "w+")
	empathyFile.write(empathyArray[0].getCSVHeader()+"\n")
	for empathy in empathyArray:
		empathyFile.write(empathy.toCSV().strip()+"\n")

	characterFile = open("experimentout/character.csv", "w+")
	characterFile.write(characterArray[0].getCSVHeader()+"\n")
	for character in characterArray:
		characterFile.write(character.toCSV().strip()+"\n")

	infoFile = open("experimentout/info.csv", "w+")
	infoFile.write(infoArray[0].getCSVHeader()+"\n")
	for info in infoArray:
		infoFile.write(info.toCSV().strip()+"\n")

	importanceFile = open("experimentout/importance.csv", "w+")
	importanceFile.write(importanceArray[0].getCSVHeader()+"\n")
	for importance in importanceArray:
		importanceFile.write(importance.toCSV().strip()+"\n")

	adviceFile = open("experimentout/advice.csv", "w+")
	adviceFile.write(adviceArray[0].getCSVHeader()+"\n")
	for advice in adviceArray:
		adviceFile.write(advice.toCSV().strip()+"\n")

	advicecharFile = open("experimentout/advicechar.csv", "w+")
	advicecharFile.write(advicecharArray[0].getCSVHeader()+"\n")
	for advicechar in advicecharArray:
		advicecharFile.write(advicechar.toCSV().strip()+"\n")

	voteadviceFile = open("experimentout/voteadvice.csv","w+")
	voteadviceFile.write(voteadviceArray[0].getCSVHeader()+"\n")
	for voteadvice in voteadviceArray:
		voteadviceFile.write(voteadvice.toCSV().strip()+"\n")


def parseXML():
	parseDilemmas()
	parseEmpathy()
	parseCharacter()
	parseInfo()
	parseImportance()
	parseAdvice()
	parseAdviceChar()
	parseVoteAdvice()
	parseOpenDilemmas()
	parseAnswerDilemmas()
	parseOpenInfos()
	parseShowAdvice()
	parseRootDilemmas()


def calculateStats():
	sumDilTimeToAnswer = 0
	sumDilTimeOpen = 0
	sumDilNeutralFeedback = 0
	sumDilPositiveFeedback = 0
	sumDilNegativeFeedback = 0
	sumDilAdviceRequested = 0
	sumDilInfosRead = 0
	sumDilInfos = 40
	sumDilTimesOpened = 0
	sumDilDirectInfo = 0
	sumDilIndirectInfo = 0

	for dil in dilemmaArray:
		sumDilTimeToAnswer += int(dil.timeToAnswer)
		sumDilTimeOpen += int(dil.timeOpen)
		sumDilInfosRead += int(dil.numberInfosRead)
		sumDilTimesOpened += int(dil.timesOpened)
		sumDilDirectInfo += int(dil.directInfos)
		sumDilIndirectInfo += int(dil.indirectInfos)


		if dil.empathyFeedback == "NEUTRAL":
			sumDilNeutralFeedback += 1
		elif dil.empathyFeedback == "POSITIVE":
			sumDilPositiveFeedback += 1
		else: 
			sumDilNegativeFeedback += 1

		if dil.adviceRequested:
			sumDilAdviceRequested += 1

		avgTimeToAnswer = sumDilTimeToAnswer / 8
		avgDilTimeOpen = sumDilTimeOpen / 8
		avgDilTimesOpened = sumDilTimesOpened / 8

	openInfoArray.sort(key=lambda x: int(x.gameTime)) #sort by gametime
	flag = 0	
	sumInfoOpen = 0
	for i in range(0,len(openInfoArray)):
		if i == len(openInfoArray)-1:
			for d in answerDilemmaArray:
				if d.dilemmaId == openInfoArray[i].dilemmaId:
					sumInfoOpen += int(d.gameTime) - int(openInfoArray[i].gameTime)
			flag += 1
		else:
			if openInfoArray[i+1].dilemmaId == openInfoArray[i].dilemmaId: #same dilemma id
				sumInfoOpen += int(openInfoArray[i+1].gameTime) - int(openInfoArray[i].gameTime)
			else: #find dilemma answer time 
				for d in answerDilemmaArray:
					if d.dilemmaId == openInfoArray[i].dilemmaId:
						sumInfoOpen += int(d.gameTime) - int(openInfoArray[i].gameTime)
				flag += 1
	avgInfoOpen = sumInfoOpen / flag

	statsFile = open("experimentout/stats.csv", "w+")
	statsFile.write("sumAdvisorMood,averageInfoOpen,averageTimeToAnswer,averageTimeDilemmaOpen,sumPosFeedback,sumNegFeedback,sumNeuFeedback,sumAdviceRequested,sumInfosRead,sumInfos,avgTimesDilemmaOpened,sumDirectInfo,sumIndirectInfo"+"\n")
	statsFile.write(str(dilemmaArray[0].sumAdvisorMood)+","+str(avgInfoOpen)+","+str(avgTimeToAnswer)+","+str(avgDilTimeOpen)+","+str(sumDilPositiveFeedback)+","+
		str(sumDilNegativeFeedback)+","+str(sumDilNeutralFeedback)+","+str(sumDilAdviceRequested)+","+str(sumDilInfosRead)+","+str(sumDilInfos)+","+str(avgDilTimesOpened)+","+
		str(sumDilDirectInfo)+","+str(sumDilIndirectInfo))




#-------   MAIN   -----------------	
if len(sys.argv)>1: 
	filePath = sys.argv[1]
	root = ET.parse(filePath).getroot();
	parseXML()
	createCSV()
	calculateStats()
	print("Finished parsing.")
else:
	print("Please enter file path. Example: >> parser.py MYFILE.xml")