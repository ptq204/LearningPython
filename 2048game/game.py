import numpy as np
import time
import os
import math
import tkinter as tk
import turtle
import random

NumBlank = 16

class Game2048():
    a = np.zeros(16, dtype = int)
    a = np.reshape(a,(4,4))
    def printBoard(self):
        for i in range(0,4):
            for j in range(0,4):
                print(self.a[i][j], end = '\t')
            print('\n')
    
    def randomInit(self):
        global NumBlank
        initNum = [2,4]
        num = random.choice(initNum)
        if(NumBlank != 0):
            while(1):
                indexCol = random.randint(0,3)
                indexRow = random.randint(0,3)
                if(self.a[indexRow][indexCol] == 0):
                    self.a[indexRow][indexCol] = num
                    break     
    
    def initBoard(self):
        self.randomInit()
    
    def PlayGame(self):
        global NumBlank
        score = 0
        tmp = 0
        while(1):
            z = input('Enter an control: ')
            if(z == 'w'):
                border = 3
                for j in range(0,4):
                    for i in range(0,4):
                        tmp = i - 1
                        if(self.a[i][j] != 0):
                            while(tmp >= 0):
                                if(self.a[tmp][j] != 0):
                                    border = tmp
                                    break
                                tmp-=1
                            if(tmp < 0):
                                border = 0
                            if(border+1 < 4):
                                if(border != i):
                                    if(self.a[border][j] == self.a[i][j]):
                                        self.a[border][j]*=2
                                        self.a[i][j] = 0
                                        score += self.a[border][j]
                                        NumBlank += 1
                                    elif(border == 0 and self.a[border][j] == 0):
                                        self.a[0][j] = self.a[i][j]
                                        self.a[i][j] = 0
                                    else:
                                        if(border+1 != i):
                                            self.a[border+1][j] = self.a[i][j]
                                            self.a[i][j] = 0
            elif(z == 's'):
                border = 3
                for j in range(3,-1,-1):
                    for i in range(3,-1,-1):
                        tmp = i + 1
                        if(self.a[i][j] != 0):
                            while(tmp < 4):
                                if(self.a[tmp][j] != 0):
                                    border = tmp
                                    break
                                tmp+=1
                            if(tmp >= 4):
                                border = 3
                            if(border-1 >= 0):
                                if(border != i):
                                    if(self.a[border][j] == self.a[i][j]):
                                        self.a[border][j]*=2
                                        self.a[i][j] = 0
                                        score += self.a[border][j]
                                        NumBlank += 1
                                    elif(border == 3 and self.a[3][j] == 0):
                                        self.a[3][j] = self.a[i][j]
                                        self.a[i][j] = 0
                                    else:
                                        if(border-1 != i):
                                            self.a[border-1][j] = self.a[i][j]
                                            self.a[i][j] = 0
            elif(z == 'a'):
                border = 3
                for i in range(0,4):
                    for j in range(0,4):
                        tmp = j - 1
                        if(self.a[i][j] != 0):
                            while(tmp >= 0):
                                if(self.a[i][tmp] != 0):
                                    border = tmp
                                    break
                                tmp-=1
                            if(tmp < 0):
                                border = 0
                            if(border+1 < 4):
                                if(border != j):
                                    if(self.a[i][border] == self.a[i][j]):
                                        self.a[i][border]*=2
                                        self.a[i][j] = 0
                                        score += self.a[i][border]
                                        NumBlank += 1
                                    elif(border == 0 and self.a[i][0] == 0):
                                        self.a[i][0] = self.a[i][j]
                                        self.a[i][j] = 0
                                    else:
                                        if(border+1 != j):
                                            self.a[i][border+1] = self.a[i][j]
                                            self.a[i][j] = 0
            else:
                border = 3
                for i in range(3,-1,-1):
                    for j in range(3,-1,-1):
                        tmp = j + 1
                        if(self.a[i][j] != 0):
                            while(tmp < 4):
                                if(self.a[i][tmp] != 0):
                                    border = tmp
                                    break
                                tmp+=1
                            if(tmp >= 4):
                                border = 3
                            if(border-1 >= 0):
                                if(border != j):
                                    if(self.a[i][border] == self.a[i][j]):
                                        self.a[i][border]*=2
                                        self.a[i][j] = 0
                                        score += self.a[i][border]
                                        NumBlank += 1
                                    elif(border == 3 and self.a[i][3] == 0):
                                        self.a[i][3] = self.a[i][j]
                                        self.a[i][j] = 0
                                    else:
                                        if(border-1 != j):
                                            self.a[i][border-1] = self.a[i][j]
                                            self.a[i][j] = 0                                
            os.system('cls')
            print('Score: ', score)
            self.randomInit()
            NumBlank -= 1
            self.printBoard()

def main():
    game = Game2048()
    game.initBoard()
    game.printBoard()
    game.PlayGame()

if __name__ == '__main__':
    main()



    