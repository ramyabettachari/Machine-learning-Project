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

def ada(iter, p,f,bound):

    threshold =[]
    for index in range(0,no_of_samples-1):
        threshold.append((x[index] + x[index + 1]) / 2)

    g_big = []
    p_list = []
    g_big2 = []
    p_list2 = []
    for value in threshold:
        h = [1] * no_of_samples
        for index,sample in enumerate(x):
            if sample < value:
                h[index] = 1
            else:
                h[index] = -1
        p_rplus1 = 0
        p_rneg1 = 0
        p_wplus1 = 0
        p_wneg1 = 0

        for index, (i, j) in enumerate(zip(h,y)):
            if i == 1 and j == 1:
                p_rplus1 += p[index]

            if i == -1 and j == -1:
                p_rneg1 += p[index]

            if i == -1 and j == 1:
                p_wplus1 += p[index]

            if i == 1 and j == -1:
                p_wneg1 += p[index]

        p_list.append([p_rplus1, p_rneg1, p_wplus1, p_wneg1])
        gbig = pow((p_rplus1 * p_wneg1),1/2) + pow((p_wplus1 * p_rneg1),1/2)
        g_big.append(gbig)

        h2 = [1] * no_of_samples
        for index,sample in enumerate(x):
            if sample < value:
                h2[index] = -1
            else:
                h2[index] = 1
        p_rplus2 = 0
        p_rneg2 = 0
        p_wplus2 = 0
        p_wneg2 = 0

        for index, (i, j) in enumerate(zip(h,y)):
            if i == 1 and j == 1:
                p_rplus2 += p[index]

            if i == -1 and j == -1:
                p_rneg2 += p[index]

            if i == -1 and j == 1:
                p_wplus2 += p[index]

            if i == 1 and j == -1:
                p_wneg2 += p[index]

        p_list2.append([p_rplus2, p_rneg2, p_wplus2, p_wneg2])
        gbig2 = pow((p_rplus2 * p_wneg2),1/2) + pow((p_wplus2 * p_rneg2),1/2)
        g_big2.append(gbig2)

    min_g_big1 = min(g_big)
    min_g_big2 = min(g_big2)
    min_g_big = min(min_g_big1, min_g_big2)
    if min_g_big1 <= min_g_big2:
        min_gbig_index = g_big.index(min_g_big1)
        best_p_list = p_list[min_gbig_index]
    else:
        min_gbig_index = g_big.index(min_g_big2)
        best_p_list = p_list2[min_gbig_index]
    best_threshold = threshold[min_gbig_index]

    if min_g_big in g_big:
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
    print ('G['+str(iter+1)+']:',min_g_big)

    weight_plus = 0.5 * math.log((best_p_list[0] + epsilon) / (best_p_list[3] + epsilon))
    weight_neg = 0.5 * math.log((best_p_list[2] + epsilon) / (best_p_list[1] + epsilon))
    print ('The weights: c'+str(iter+1)+'+',weight_plus,'c'+str(iter+1)+'-', weight_neg)

    prenorm_p = []
    z = 0
    for i,j,k in zip(o,p,y):
        if i == 1:
            prenorm_p.append(j * math.exp(-k*weight_plus))
            z += j * math.exp(-k * weight_plus)
        else:
            prenorm_p.append(j * math.exp(-k*weight_neg))
            z += j * math.exp(-k * weight_neg)


    print ('The probabilities normalization factor Z['+str(iter+1)+']:',z)

    new_p = []
    new_p1 = []
    for i in prenorm_p:
        new_p.append(i / z)
    print('The probabilities after normalization:',new_p)

    new_o = []
    f1 = [0]*no_of_samples
    for index, i in enumerate(o):
        if i == 1:
            f[index] += weight_plus
            if f[index] < 0:
                new_o.append(-1)
            else:
                new_o.append(1)
        else:
            f[index] += weight_neg
            if f[index] < 0:
                new_o.append(-1)
            else:
                new_o.append(1)
    print('The values f(x):', f)

    sum = 0
    for i,j in zip(y,new_o):
        if i != j:
            sum += 1
    error_boosted = sum / no_of_samples
    print('The error of the boosted classifier:', error_boosted)

    bound *= z
    print('The bound on error of the boosted classifier:', bound)
    print('--------------------------------------------------------------------------------------')
    return new_p,f,bound

f = [0]*no_of_samples
bound = 1
for iter in range(0,no_of_iterations):
    p,f,bound = ada(iter, p,f,bound)



