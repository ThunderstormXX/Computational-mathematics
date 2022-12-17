from matplotlib import pylab as plt
from copy import deepcopy
import sys
import math

def pivotize(mat_a, x):
    """
    Путем обмена строк расположить наибольшие элементы на диагонали
    """
    mat_a = deepcopy(mat_a)
    size = len(mat_a)
    row = max(range(x, size), key=lambda i: abs(mat_a[i][x]))
    if x != row:
        mat_a[x], mat_a[row] = mat_a[row], mat_a[x]
    return mat_a

def invert(mat_a):
    """
    Обращение матрицы методом Гаусса-Жордана
    """
    mat_a = deepcopy(mat_a)
    n = len(mat_a)

    # Дополнить матрицу справа единичной матрицей
    for i in range(n):
        mat_a[i] += [int(i == j) for j in range(n)]

    # Прямой ход
    for x in range(n):
        mat_a = pivotize(mat_a, x)
        for i in range(x + 1, n):
            coefficient = mat_a[i][x] / mat_a[x][x]
            for j in range(x, n * 2):
                mat_a[i][j] -= coefficient * mat_a[x][j]

    # Обратный ход
    for x in reversed(range(n)):
        for i in reversed(range(x)):
            coefficient = mat_a[i][x] / mat_a[x][x]
            for j in reversed(range(n * 2)):
                mat_a[i][j] -= coefficient * mat_a[x][j]

    # Разделить строки на ведущие элементы
    for i in range(n):
        denominator = mat_a[i][i]
        for j in range(n * 2):
            mat_a[i][j] /= denominator

    # Оставить только правую часть матрицы
    for i in range(n):
        mat_a[i] = mat_a[i][n:]

    return mat_a


def transpose(matrix):
    res = []
    for i in range(len(matrix[0])):
        temp = []
        for j in range(len(matrix)):
            temp.append(matrix[j][i])
        res.append(temp)
    return res

def matrixmult (A, B):
    if type(B[0]) is not int and type(B[0]) is not float :
        C = [[0 for row in range(len(B[0]))] for col in range(len(A))]
        # print(C)
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    C[i][j] += A[i][k]*B[k][j]
        return C
    else :
        C =[0 for row in range(len(A))]
        for i in range(len(A)):
            for k in range(len(A[0])):
                C[i] += A[i][k]*B[k]
        return C


def solver(X,Y):
    X_T = transpose(X)
    Z = matrixmult(X_T,X)        
    Z1 = invert(transpose(Z))
    Z2 = matrixmult(Z1,X_T) 
    w = matrixmult(Z2,Y)

    #Подсчет ошибки
    z = matrixmult(X,w)
    st_error =  float(0)
    for i in range(len(z)):
        st_error += (z[i]-Y[i])*(z[i]-Y[i])
    if st_error < 1e-5 :
        st_error = 0
    st_error = pow(st_error,1//2)
    
    #Выводм ногочлена (Здесь настраиваем точность с которой мы хотим выводить ответ)
    print(w)
    if abs(w[0]) < 1e-3:
        w[0] = 0
    result = str(round(w[0],4))
    for i in range(1,len(w)):
        #print(w[i] ," ", 1e-3 , " " ,w[i] < 1e-3)
        if abs(w[i]) < 1e-3:
            w[i] = 0
        result += " " + str(round(w[i],4))+"x^"+str(i)  
    
    
    return (result, st_error,w)

def func(x_res,w):
    res = []
    for j in range(len(x_res)):
        res.append(0)
        for i in range(len(w)):
            res[j]+= w[i]*pow(x_res[j],i)
    return res


if __name__ == '__main__' :
    print("Введите число точек :")
    n = int(input())
    print("Введите максимальную степень  приближающего многочлена :")
    k = int(input()) + 1
    print("Введите n точек в формате : x_1 y_1  \\n x_2 y_2 ... \\n x_n y_n")
    
    X = []
    Y = []
    for i in range(n) :
        x , y = input().split()
        Y.append(float(y))
        features = [pow(float(x),i) for i in range(k)]
        X.append(features)
    x_label = []
    y_label = Y
    for i in range(n):
        x_label.append(X[i][1])
    
    if n < k :
        #solve linear equation
        print("Выберите k поменьше,ведь для идеальной аппроксимации достаточно многочлена меньшей степени")
        pass
    else :
        (result_answer , standart_error , w) = solver(X,Y)
        print("Среднеквдартичная ошибка при приближении многочлена : ",standart_error)
        print("Многочлен : " ,result_answer)
        


        plt.figure(figsize=(10, 8))
        plt.scatter(x_label, y_label, color = 'red')
       
        x_min = min(x_label)
        x_max = max(x_label)
        x_res = []
        for i in range(100):
            x_res.append(x_min + i*(x_max-x_min)/100)
        y_res = func(x_res,w)
        #plt.scatter(x_res, y_res, color = 'black')
        plt.plot(x_res, y_res , '--', color = 'black')
        plt.title(u"")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("9.jpg")
        plt.show()