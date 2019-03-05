import nltk
#nltk.download('stopwords')
#nltk.download('punkt')  # For using word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer    # For lemmatization, not used it in the code yet
import numpy as np
import csv
import math

class refineSpamData :
    
    def __init__(self , csvFileName) :
        self.csvFileName            = csvFileName
        self.listOfDistinctWords    = []
        
    def start(self) :
        self.getStopWords()
        self.readCSV()
        self.countOccurrences()
        self.calcProbOfSpamOrHam()
        self.initializeConfusionMatrix()
        self.useBayesRuleOnRemainingData()
        self.getConfusionMatrixMeasures()
        
    def initializeConfusionMatrix(self) :
        # As this is a binary classification problem, we assign 0 to Spam, 1 to Ham
        self.sizeOfConfusionMatrix  = 2
        self.confusionMatrix        = np.zeros( [2 , 2] )
        
    def getConfusionMatrixMeasures(self) :
        self.getAccuracy()
        self.getPrecision()
        self.getRecall()
        self.printConfusionMatrixMeasures()
        
    def printConfusionMatrixMeasures(self) :
        print("Confusion Matrix (0 for Spam and 1 for Ham) : " )
        print( self.confusionMatrix )
        print("Accuracy     : " , self.accuracy)
        print("SPAM")
        print("Precision    : " , self.precisionForClassSpam)
        print("Recall       : " , self.recallForClassSpam)
        print("HAM")
        print("Precision    : " , self.precisionForClassHam)
        print("Recall       : " , self.recallForClassHam)
        
    def getAccuracy(self) :
        noOfPtsClassifiedCorrectly  = sum([self.confusionMatrix[i][i] for i in range(self.sizeOfConfusionMatrix)])
        totalNoOfPredictions        = sum(sum(self.confusionMatrix))
        self.accuracy               = noOfPtsClassifiedCorrectly/totalNoOfPredictions
        
    def getPrecision(self) :
        self.precisionForClassSpam  = self.getPrecisionForNthClass( N=0 )
        self.precisionForClassHam   = self.getPrecisionForNthClass( N=1 )
        
    def getPrecisionForNthClass(self , N) :
        noOfTruePositives           = self.confusionMatrix[N][N]
        noOfPtsClassifiedAsPositive = sum( self.confusionMatrix[N , :] )
        return noOfTruePositives/noOfPtsClassifiedAsPositive
    
    def getRecall(self) :
        self.recallForClassSpam = self.getRecallForNthClass( N = 0 )
        self.recallForClassHam  = self.getRecallForNthClass( N = 1 )
        
    def getRecallForNthClass(self, N) :
        noOfTruePositives           = self.confusionMatrix[N][N]
        noOfPtsActuallyPositive     = sum( self.confusionMatrix[: , N] )
        return noOfTruePositives/noOfPtsActuallyPositive
        
    def calcProbOfSpamOrHam(self) :
        self.priorProbOfSpam = self.noOfSpamMessages/self.totalNoOfMessages
        self.priorProbOfHam  = self.noOfHamMessages/self.totalNoOfMessages
        
    def useBayesRuleOnRemainingData(self) :
        for i in range( self.noOfDataPointsToConsiderForProbCalc , self.totalNoOfMessages + 1 ) :
            messageString                   = self.data[i][1]
            stringStrippedOfStopWordsAsList = self.getUsableMessageWordsAsList( messageString )
            # not the whole bayes expression, just one of the terms
            probOfSpamGivenMessageString    = self.priorProbOfSpam*self.getProductOfPriorProbabilitiesGivenSpam(stringStrippedOfStopWordsAsList)
            probOfHamGivenMessageString     = self.priorProbOfHam*self.getProductOfPriorProbabilitiesGivenHam(stringStrippedOfStopWordsAsList)
            spamOrHamWord                   = self.data[i][0]
            self.updateConfusionMatrix(probOfSpamGivenMessageString , probOfHamGivenMessageString , spamOrHamWord)
#        print( self.confusionMatrix )

    def updateConfusionMatrix(self , probOfSpamGivenMessageString , probOfHamGivenMessageString , spamOrHamWord) :
        if probOfSpamGivenMessageString > probOfHamGivenMessageString :
            if spamOrHamWord.strip() == "spam" :
                self.confusionMatrix[0][0]  += 1
            else :
                self.confusionMatrix[0][1]  += 1
        elif probOfHamGivenMessageString > probOfSpamGivenMessageString :
            if spamOrHamWord.strip() == "ham" :
                self.confusionMatrix[1][1]  += 1
            else :
                self.confusionMatrix[1][0]  += 1
            
    def getProductOfPriorProbabilitiesGivenHam(self , stringStrippedOfStopWordsAsList) :
        product = 1e50
        for word in stringStrippedOfStopWordsAsList :
            if word in self.dictForCountingWordOccurrenceAndHam :
                prob = self.dictForCountingWordOccurrenceAndHam[word]/self.noOfHamMessages
            else :
                prob = 0.000001
#                prob = 1
            product *= prob
        return product
        
    def getProductOfPriorProbabilitiesGivenSpam(self , stringStrippedOfStopWordsAsList) :
        product = 1e50
        for word in stringStrippedOfStopWordsAsList :
            if word in self.dictForCountingWordOccurrenceAndSpam :
                prob = self.dictForCountingWordOccurrenceAndSpam[word]/self.noOfSpamMessages
            else :
                prob = 0.000001
#                prob = 1
            product *= prob
        return product
        
    def countOccurrences(self) :
        self.totalNoOfMessages  = len(self.data)-1
        self.noOfSpamMessages   = 0
        self.noOfHamMessages    = 0
        self.dictForCountingWordOccurrenceAndSpam   = {}
        self.dictForCountingWordOccurrenceAndHam    = {}
        self.getNoOfDataPointsToConsiderForProbCalc()
        for i in range( 1 , self.noOfDataPointsToConsiderForProbCalc ) :
            spamOrHamWord                   = self.data[i][0]
            self.countNoOfSpamAndHam( spamOrHamWord )
            messageString               = self.data[i][1]
            usableMessageWordsAsList    = self.getUsableMessageWordsAsList( messageString )
            self.countWordOccurrenceGivenSpamOrHam( spamOrHamWord , usableMessageWordsAsList )
            
    def getUsableMessageWordsAsList(self , messageString) :
        stringStrippedOfStopWordsAsList                     = self.removeStopWords( messageString )
        stringStrippedOfStopWordsAndSpecialCharactersAsList = self.removeSpecialCharacters( stringStrippedOfStopWordsAsList )
        finalString = self.removeDuplicateWords( stringStrippedOfStopWordsAndSpecialCharactersAsList )
        return finalString
    
    def removeDuplicateWords(self , listOfWordsAsString) :
        return list(set(listOfWordsAsString))    # Remove duplicates from list of words from each message
        
    def removeSpecialCharacters(self , listOfWordsAndSpecialCharacters) :
        return [ w for w in listOfWordsAndSpecialCharacters if w.isalnum() ]
    
    def getStopWords(self) :
        self.stopWords = set(stopwords.words('english'))

    def readCSV(self) :
        fileHandleReadType      = open(self.csvFileName , "rU")
        reader                  = csv.reader(fileHandleReadType , delimiter = ",")
        self.data               = list(reader)
        fileHandleReadType.close()
        
    # What if a word occurs multiple times in a single message
    def countWordOccurrenceGivenSpamOrHam(self , spamOrHamWord , stringStrippedOfStopWordsAsList) :
        if spamOrHamWord.strip() == "spam" :
            for word in stringStrippedOfStopWordsAsList :
                if word in self.dictForCountingWordOccurrenceAndSpam :
                    self.dictForCountingWordOccurrenceAndSpam[word] += 1
                else :
                    self.dictForCountingWordOccurrenceAndSpam[word] = 1
        else :
            for word in stringStrippedOfStopWordsAsList :
                if word in self.dictForCountingWordOccurrenceAndHam :
                    self.dictForCountingWordOccurrenceAndHam[word] += 1
                else :
                    self.dictForCountingWordOccurrenceAndHam[word] = 1
        
    def getNoOfDataPointsToConsiderForProbCalc(self) :
        self.noOfDataPointsToConsiderForProbCalc = math.floor( 0.7*self.totalNoOfMessages )
        
    def removeStopWords(self , messageString) :
        stringAsListOfWordsAndSpecialCharacters = word_tokenize( messageString )
        # converting to lower case so that probabilities are captured properly
        filteredString = [w.lower() for w in stringAsListOfWordsAndSpecialCharacters if not w.lower() in self.stopWords]
#        filteredString = " ".join( filteredString )    # To get a sentence out of a string of words
        return filteredString
            
    def countNoOfSpamAndHam(self , spamOrHamWord) :
        if spamOrHamWord.strip() == "spam" :
            self.noOfSpamMessages   += 1
        else :
            self.noOfHamMessages    += 1
    
csvFileName         = "spam.csv"
refineSpamDataObj   = refineSpamData( csvFileName )
refineSpamDataObj.start()