import math
Inputfile = input('Enter the name of the input file:')

with open(Inputfile) as input:
    numbers = input.read().split()

no_of_iterations = int(numbers[0])
no_of_samples = int(numbers[1])
epsilon = float(numbers[2])
x = numbers[3:3+no_of_samples]
x = [float(i) for i in x]
y = numbers[3+no_of_samples:3+2*no_of_samples]
y = [int(i) for i in y]
p = numbers[3+2*no_of_samples:3+3*no_of_samples]
p = [float(i) for i in p]

def ada(iter, p,f,f_symbol,bound):

    threshold =[]
    for index in range(0,no_of_samples-1):
        threshold.append((x[index]+x[index+1])/2)

    error = []
    for value in threshold:
        h = [1] * no_of_samples
        for index, x_sample in enumerate(x):
            if x_sample < value:
                h[index] = 1
            else:
                h[index] = -1
        err1 = 0
        err2 = 0
        for index, (i, j) in enumerate(zip(y,h)):
            if i != j:
                err1 += p[index]
            else:
                err2 += p[index]
        error.append(err1)
        error.append(err2)

    min_error = min(error)
    min_error_index = error.index(min_error)
    best_threshold = threshold[int(min_error_index/2)]
    if min_error_index % 2 == 0:
        best_threshold_sign = 1
    else:
        best_threshold_sign = -1
    if best_threshold_sign == 1:
        symbol = '<'
    else:
        symbol = '>'
    o = []
    for i in x:
        if i < best_threshold:
            o.append(best_threshold_sign)
        else:
            o.append(-best_threshold_sign)
    print ('The selected weak classifier: I(x'+str(symbol)+str(best_threshold)+')')
    print ('The error of h['+str(iter+1)+']:',min_error)

    weight = 0.5 * math.log((1-min_error)/min_error)
    print ('The weight of h['+str(iter+1)+']:',weight)

    prenorm_p = []
    q_wrong = math.exp(weight)
    q_right = math.exp(-weight)
    for i,j,k in zip(y, o, p):
        if i != j:
            prenorm_p.append(q_wrong * k)
        else:
            prenorm_p.append(q_right * k)

    z = 0
    for i,j,k in zip(y, o, p):
        if i != j:
            z += k * q_wrong
        else:
            z += k * q_right
    print ('The probabilities normalization factor:',z)

    new_p = []
    for i in prenorm_p:
        new_p.append(i / z)
    print('The probabilities after normalization:',new_p)

    new_o = []
    for index,i in enumerate(o):
        f[index] += weight * i
        if f[index] < 0:
            new_o.append(-1)
        else:
            new_o.append(1)

    if iter == 0:
        f_symbol = f_symbol+str(weight)+'I(x'+symbol+str(best_threshold)+')'
    else:
        f_symbol = f_symbol+'+' + str(weight) + 'I(x' + symbol + str(best_threshold) + ')'
    print('The boosted classifier:', f_symbol)

    sum = 0
    for i,j in zip(y,new_o):
        if i != j:
            sum += 1
    error_boosted = sum / no_of_samples
    print('The error of the boosted classifier:', error_boosted)

    bound *= z
    print('The bound on E['+str(iter+1)+']:', bound)
    print('--------------------------------------------------------------------------------------')
    return new_p,f,f_symbol,bound

f = [0]*no_of_samples
f_symbol = ''
bound = 1
for iter in range(0,no_of_iterations):
    p,f,f_symbol,bound = ada(iter, p,f,f_symbol,bound)



