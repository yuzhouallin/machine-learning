import csv
import math



def read_data_set(csv_file):
    matrix = []
    with open(csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        x = 0
        for row in reader:
            x = x + 1
            y = len(row)
            matrix.append(row)
    #print matrix
    #print x , y
    return matrix, x, y

def calc_mu_sigma(matrix, x, y):
    mu = {}
    sigma = {}
    for j in range(0, y):
        mu[j] = 0
        for i in range(0, x):
            mu[j] = mu[j] + float(matrix[i][j])
        mu[j] = float(mu[j]) / float(x)
        sigma[j] = 0
        for i in range(0, x):
            sigma[j] = sigma[j] + math.pow(float(matrix[i][j]) - mu[j], 2)

        sigma[j] = float(sigma[j]) / float(x)
    #print mu
    #print sigma
    return mu, sigma, x, y

def calc_gaussian_p(matrix, mu, sigma, x, y):
    p = {}
    for i in range(0, x):
        try:
            p[i] = 1
            for j in range(0, y):
                num_01 = 1.0/(math.sqrt(2*math.pi) * math.sqrt(sigma[j]))
                num_02 = (-1.0 * math.pow(float(matrix[i][j])-mu[j],2) )/ (2 * sigma[j])
                p[i] = p[i] * num_01 * math.exp(num_02)
        except Exception as pex:
            p[i] = 0
    #print p
    return p


matrix, x, y = read_data_set('normal.csv')
mu, sigma, x, y = calc_mu_sigma(matrix, x, y)
p = calc_gaussian_p(matrix, mu, sigma, x, y)
print p
matrix2, xx, yy = read_data_set('abnormal.csv')
p2 = calc_gaussian_p(matrix2, mu, sigma, xx, yy)
print p2