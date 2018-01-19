#!/usr/bin/python
#
import xml.etree.ElementTree as ET
import sys
import os

#define arrays where objects will be saved
dilemmaArray = []
infoArray = []
stylesArray = []
empathygroupArray = []

#dilema class
class Dilemma(object):
	def __init__(self, logToken, id, openedDuration, delegateAttempted, delegateCount, adviceCount, answerType):
		self.logToken = logToken
		self.id = id
		self.openedDuration = openedDuration
		self.delegateAttempted = delegateAttempted
		self.delegateCount = delegateCount
		self.adviceCount = adviceCount
		self.answerType = answerType

	def getCSVHeader(self):
		return "log token,id,opened duration,delegate attempted,delegate count,advice count,answer type"

	def toCSV(self):
		return self.logToken+","+self.id+","+self.openedDuration+","+self.delegateAttempted+","+self.delegateCount+","+self.adviceCount+","+self.answerType

#info class
class Info(object):
	def __init__(self, id, dilemmaId, hasBeenOpened, openedCount, importanceType):
		self.id = id
		self.dilemmaId = dilemmaId
		self.hasBeenOpened = hasBeenOpened
		self.openedCount = openedCount
		self.importanceType = importanceType

	def getCSVHeader(self):
		return "id,dilemma id,has been opened,opened count,importance type"

	def toCSV(self):
		return self.id+","+self.dilemmaId+","+self.hasBeenOpened+","+self.openedCount+","+self.importanceType

#style class
class ScoredStyle(object):
	def __init__(self, dilemmaId, style):
		self.dilemmaId = dilemmaId
		self.style = style

	def getCSVHeader(self):
		return "dilemma id,style"

	def toCSV(self):
		return self.dilemmaId+","+self.style

#empathy group class
class Empathygroup(object):
	def __init__(self, dilemmaId, group, mood):
		self.dilemmaId = dilemmaId
		self.group = group
		self.mood = mood

	def getCSVHeader(self):
		return "dilemma id,group,mood"

	def toCSV(self):
		return self.dilemmaId+","+self.group+","+self.mood


def getUsername():
	playdata = root.find('playdata')
	return playdata.attrib.get('username')

def getLogToken(root):
	playdata = root.find('playdata')
	return playdata.attrib.get('logtoken')

def getStartTime(root):
	playdata = root.find('playdata')
	return playdata.attrib.get('gamestart')

def getEndtime(root):
	playdata = root.find('playdata')
	return playdata.attrib.get('gameend')

#parse infos data from XML into array of Info objects
def parseInfos(dilemma):
	dilemmaId = dilemma.find('id').text.strip()
	for infos in dilemma.iter('infos'):
		for info in infos.iter('info'):

			id = info.find('id').text.strip()
			hasBeenOpened = info.attrib.get('has-been-opened')
			openedCount = info.attrib.get('opened-count')
			importanceType = info.attrib.get('importance-type')
			i = Info(id, dilemmaId, hasBeenOpened, openedCount, importanceType)
			infoArray.append(i)

#parse scored styles data from XML into array of ScoredStyles objects
def parseScoredStyles(dilemma):
	dilemmaId = dilemma.find('id').text.strip()
	for scoredstyles in dilemma.iter('scoredstyles'):
		for scoredstyle in scoredstyles.iter('scoredstyle'):

			style = scoredstyle.text
			s = ScoredStyle(dilemmaId, style)
			stylesArray.append(s)

#parse empathy groups data from XML into array of EmpathyGroup objects
def parseEmpathyGroups(dilemma):
	dilemmaId = dilemma.find('id').text.strip()
	for empathygroups in dilemma.iter('empathygroups'):
		for empathygroup in empathygroups.iter('empathygroup'):

			group = empathygroup.text
			mood =empathygroup.attrib.get('mood')
			e = Empathygroup(dilemmaId, group, mood)
			empathygroupArray.append(e)

#parse dilemmas data from XML into array of Dilemma objects
def parseDilemmas():
	for results in root.iter('results'):
		for dilemmas in results.iter('dilemmas'):
			for dilemma in dilemmas.iter('dilemma'):

				logToken = getLogToken(root)
				dilemmaId = dilemma.find('id').text.strip()
				openedDuration = dilemma.attrib.get('opened-duration')
				delegateAttempted = dilemma.attrib.get('delegate-attempted')
				delegateAttemptCount = dilemma.attrib.get('delegate-attempt-count')
				adviceCount = dilemma.attrib.get('advice-count')
				answerType = dilemma.attrib.get('answer-type')

				d = Dilemma(logToken, dilemmaId, openedDuration, delegateAttempted, delegateAttemptCount, adviceCount, answerType);
				dilemmaArray.append(d)

				parseInfos(dilemma)

				parseScoredStyles(dilemma)

				parseEmpathyGroups(dilemma)

#start parsing XML
def parseXML():
	parseDilemmas()
	
#create CSV from object arrays
def createCSV():
	if not os.path.exists("output"): 
		os.makedirs("output")

	dilemmaFile = open("output/dilemmas.csv", "w+")
	dilemmaFile.write(dilemmaArray[0].getCSVHeader()+"\n")
	for dilemma in dilemmaArray:
		#dilemmaFile.write(dilemma.toCSV().encode("utf-8").strip().decode()+"\n")
		dilemmaFile.write(dilemma.toCSV().strip()+"\n")

	infoFile = open("output/infos.csv", "w+")
	infoFile.write(infoArray[0].getCSVHeader()+"\n")
	for info in infoArray:
		#infoFile.write(info.toCSV().encode("utf-8").strip().decode()+"\n")
		infoFile.write(info.toCSV().strip()+"\n")

	scoredStyleFile = open("output/scoredstyles.csv", "w+")
	scoredStyleFile.write(stylesArray[0].getCSVHeader()+"\n")
	for style in stylesArray:
		#scoredStyleFile.write(style.toCSV().encode("utf-8").strip().decode()+"\n")
		scoredStyleFile.write(style.toCSV().strip()+"\n")

	empathygroupFile = open("output/empathygroups.csv", "w+")
	empathygroupFile.write(empathygroupArray[0].getCSVHeader()+"\n")
	for empathygroup in empathygroupArray:
		#empathygroupFile.write(empathygroup.toCSV().encode("utf-8").strip().decode()+"\n")
		empathygroupFile.write(empathygroup.toCSV().strip()+"\n")


#-------   MAIN   -----------------	
if len(sys.argv)>1: 
	filePath = sys.argv[1]
	root = ET.parse(filePath).getroot();
	parseXML()
	createCSV()
	print("Finished parsing.")
else:
	print("Please enter file path. Example: >> parser.py MYFILE.xml")



