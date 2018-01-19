
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

class Dilemma(object):
	def __init__(self, gameId, dilemmaId, timeOpen, answerType):
		self.gameId = gameId
		self.dilemmaId = dilemmaId
		self.timeOpen = timeOpen
		self.answerType = answerType

	def getCSVHeader(self):
		return "gameId, dilemmaId,timeOpen,answerType"

	def toCSV(self):
		return self.gameId+","+self.dilemmaId+","+self.timeOpen+","+self.answerType

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
		self.askedinfo = askedinfo
		self.advMood = advMood

	def getCSVHeader(self):
		return "gameId,advisor,askedinfo,advMood"

	def toCSV(self):
		return self.gameId+","+self.advisor+","+self.askedinfo+","+self.advMood

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
        self.gameid = gameid
        self.dilemmaid = dilemmaid
        self.adviceid = adviceid
        self.importancetype = importancetype
        self.importancemoment = importancemoment
        self.time = time
		
    def getCSVHeader(self):
        return "gameId,dilemmaId,adviceid,importancetype,importancemoment,importancetime"

    def toCSV(self):
        return self.gameid+","+self.dilemmaid+","+self.adviceid+","+self.importancetype+","+self.importancemoment+","+self.time

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

#-------   MAIN   -----------------	
if len(sys.argv)>1: 
	filePath = sys.argv[1]
	root = ET.parse(filePath).getroot();
	parseXML()
	createCSV()
	print("Finished parsing.")
else:
	print("Please enter file path. Example: >> parser.py MYFILE.xml")