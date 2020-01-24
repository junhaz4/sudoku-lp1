import os
import numpy as np 
import pandas as pd 
import scipy.sparse as scs 
import scipy.linalg as scl 
import scipy.optimize as sco 
import matplotlib.pylab as plt
from numpy.linalg import norm 

class StateInputError:
    pass

class Sudoku:
    def __init__(self, sustring):
        if type(sustring) is not str:
            raise StateInputError(f'Input {sustring} is not a string')
        else:
            if len(sustring) != 81:
                raise StateInputError(f'Input {sustring} doesn\'t have legal length(81)')
        
        self.string = sustring
        self.fill = []
        for i in range(len(sustring)):
            if sustring[i] == '0':
                self.fill.append(0)
            else:
                self.fill.append(1)
        self.fill = np.array(self.fill)
        
        self.state = []
        for i in range(len(sustring)):
            temp = [0,0,0,0,0,0,0,0,0]
            if int(sustring[i]) != 0:
                temp[int(sustring[i])-1] = 1
            self.state.extend(temp)
            
        self.state = np.array(self.state)
    
    def get_fill_pos(self):
        return self.fill
    
    
    def binary_state(self):
        return self.state
    
    def real_state(self):
        real = []
        for i in range(len(self.string)):
            real.append(int(self.string[i]))
        return np.array(real).reshape(9,9)
    
    def _row_constraint(self):
        arr0 = np.zeros([9,9])
        arr1 = np.identity(9)
        temp0 = arr0
        temp1 = arr1
        for _ in range(1,9):
            arr0 = np.append(arr0,temp0,axis=1)
            arr1 = np.append(arr1,temp1,axis=1)
        rc = arr1
        for i in range(1,9):
            rc = np.append(rc,arr0,axis=1)
        for i in range(1,9):
            temp = arr0
            for j in range(1,9):
                if j==i:
                    temp = np.append(temp,arr1,axis=1)
                else:
                    temp = np.append(temp,arr0,axis=1)
            rc = np.append(rc,temp,axis=0)
        return rc
    
    def _col_constraint(self):
        block = np.append(np.identity(9),np.zeros([9,72]),axis=1)
        line1 = block
        for _ in range(1,9):
            line1 = np.append(line1,block,axis=1)
        cc = line1
        for i in range(1,9):
            temp = np.append(line1[:,729-9*i:],line1[:,:729-9*i],axis=1)
            cc = np.append(cc,temp,axis=0)
        return cc
            
    def _box_constraint(self):
        bc = np.zeros([9,729])
        for i in range(3):
            for j in range(3):
                ini = np.zeros([9,9])
                ini[3*i:3*(i+1),3*j:3*(j+1)] = 1
                ini = ini.flatten()
                bc1 = np.zeros([9,9])
                for k in ini:
                    if k == 1:
                        temp = np.identity(9)
                    else:
                        temp = np.zeros([9,9])
                    bc1 = np.append(bc1,temp,axis=1)   
                bc = np.append(bc,bc1[:,9::],axis=0)
        return bc[9:,:]
                
    def _cell_constraint(self):
        cc = []
        for i in range(len(self.string)):
            if int(self.string[i]) == 0:
                temp = []
                for j in range(i):
                    temp.extend([0,0,0,0,0,0,0,0,0])
                temp.extend([1,1,1,1,1,1,1,1,1])
                for j in range(80 - i):
                    temp.extend([0,0,0,0,0,0,0,0,0])
                cc.append(temp)
        return np.array(cc)
    
    def _clue_constraint(self):
        cc = []
        for i in range(len(self.string)):
            if int(self.string[i]) != 0:
                temp = []
                for j in range(i):
                    temp.extend([0,0,0,0,0,0,0,0,0])
                temp1 = [0,0,0,0,0,0,0,0,0]
                temp1[int(self.string[i])-1] = 1
                temp.extend(temp1)
                for j in range(80 - i):
                    temp.extend([0,0,0,0,0,0,0,0,0])
                cc.append(temp)
        return np.array(cc)
    
    def get_A(self):
        a = self._row_constraint()
        a = np.append(a,self._col_constraint(),axis = 0)
        a = np.append(a,self._box_constraint(),axis = 0)
        a = np.append(a,self._cell_constraint(),axis = 0)
        a = np.append(a,self._clue_constraint(),axis = 0)
        return a

def weighted_LP1(A, L = 10, x1 = np.zeros(729) , x2 = np.zeros(729), tol = 1e-10):
    x = x1-x2
    w = np.zeros(len(x))
    for i in range(1, L):
        eps = 0.5 #np.random.normal()
        for j in range(len(w)):
            w[j]= 1/(abs(x[j])**(i-1)+eps)
        c=np.block([w,w])
        A_eq=np.block([A, -A])
        b_eq = np.ones(len(A))   
        G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],\
                         [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
        h = np.zeros(A.shape[1]*2)
        xs = sco.linprog(c,G, h, A_eq, b_eq ,method='interior-point', bounds = (0,None))        
        xs = xs.x
        half  = int(len(xs)/2)
        xs0 = xs[:half]
        xs1 = xs[half:]     
        x_new = xs0-xs1
        if norm(x_new - x) < tol:
            break
        else:
            x = x_new
    return x_new


def dective(answer):
    l = 0
    repeat = []
    for i in range(9):
        for j in range(9):
            for k in range(j+1,9):
                if answer[i][j] == answer[i][k]:
                    repeat.extend([(i,j),(i,k)])
                    l = 1
                if answer[j][i] == answer[k][i]:
                    repeat.extend([(j,i),(k,i)])
                    l = 1
                if answer[(i//3)*3 + j//3][(i%3)*3 + j%3] == answer[(i//3)*3 + k//3][(i%3)*3 + k%3]:
                    repeat.extend([((i//3)*3 + j//3,(i%3)*3 + j%3),((i//3)*3 + k//3,(i%3)*3 + k%3)])
                    l = 1
                                  
    for x,y in repeat:
        answer[x][y] = 0
    return answer,l

def to_str(answer):
    return ''.join([''.join([str(k) for k in i]) for i in answer])

def rand_pick(answer,fill):
    while True:
        pos = np.random.randint(9, size = (1,2))[0]
        x = pos[0]
        y = pos[1]
        if fill[y+9*x] == 0 and answer[x][y] != 0:
            break
    return x,y

def repeat(newa):

    s = to_str(newa)
    s = Sudoku(s)
    A = s.get_A()
    x = weighted_LP1(A)
    z = np.reshape(x, (81, 9))
    answer = np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9) )

    return answer

def compare(a1,a2):
    result = []
    for i in range(9):
        for j in range(9):
            if a1[i][j] == a2[i][j]:
                result.append(0)
            else:
                result.append(1)
    return np.array(result).reshape(9,9)

def fourth_solver_constructor(rt,a1,a2,fill):
    c1 = compare(a1,a2)
    # random pick again
    if np.sum(c1) == 0:
        return rt
    else:
        i = 0
        while True:
            pos = np.random.randint(9, size = (1,2))[0]
            x = pos[0]
            y = pos[1]
            if fill[y+9*x] == 0 and a1[x][y] != 0 and c1[x][y] != 0:
                break
            i += 1
            if i > 300:
                break
        rt[x][y] = a1[x][y]
        return rt
    
def fifth_solver_constructor(rt,a1,a2,a3,fill):
    c1 = compare(a1,a2)
    c2 = compare(a2,a3)
    if np.sum(c1) != 0 and np.sum(c2) != 0:
        k = 0
        while k < 100:
            pos = np.random.randint(9, size = (1,2))[0]
            x = pos[0]
            y = pos[1]
            if fill[y+9*x] == 0 and a1[x][y] == 0 and c1[x][y] == 0 and c2[x][y] == 0:
                break
            k += 1
        print(a1[x][y],a2[x][y],a3[x][y])
        rt[x][y] = a1[x][y]
    return rt


def solver(quiz):
    record = []
    s = Sudoku(quiz)
    A = s.get_A()
    x = weighted_LP1(A)
    z = np.reshape(x, (81, 9))
    #print('problem',s.real_state(),sep = '\n')
    answer = np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9))
    record.append(answer)
    newa1,l1 = dective(answer)
    if l1 == 0:
        pass
    else:
        #first solver
        #print('go on to second')
        answer = repeat(newa1)
        record.append(answer)
        newa2,l2 = dective(answer)
        if l2 == 0:
            pass
        else:
            #second solver
            #print('to third')
            answer = repeat(newa1)
            record.append(answer)
            newa3,l3 = dective(answer)
            if l3 == 0:
                pass
            else:
                #third solver
                #print('to fourth')
                xp,yp = rand_pick(dective(answer)[0],s.get_fill_pos())
                new_state = s.real_state()
        
                new_state[xp][yp] = answer[xp][yp]
                #print('new problem',new_state,sep = '\n')
                answer = repeat(new_state)
                record.append(answer)
                newa4,l4 = dective(answer)
                
                if l4 == 0:
                    pass
                else:
                    #fourth solver
                    #print('to fifth')
                    new_state = s.real_state()
                    a1 = dective(record[0])[0]
                    a2 = dective(record[-1])[0]
                    newP = fourth_solver_constructor(new_state,a1,a2,s.get_fill_pos())
                    answer = repeat(newP)
                    record.append(answer)
                    '''
                    vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
                    the following part is the fifth solver which is believed not to work
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    
                    newa5,l5 = dective(answer)
                    if l5 == 0:
                        pass
                    else:
                        print('to sixth')
                        new_state = s.real_state()
                        a1 = dective(record[0])[0]
                        a2 = dective(record[-2])[0]
                        a3 = dective(record[-1])[0]
                        newP = fifth_solver_constructor(new_state,a1,a2,a3,s.get_fill_pos())
                        answer = repeat(newP)
                        record.append(answer)
                    '''
    return to_str(answer)
