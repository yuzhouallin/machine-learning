__author__ = "Yu Zhou (yuzhou2@cisco.com)"

import csv
import math
from datetime import datetime

def split_data_set(csv_file):
    with open(csv_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        nx = 0
        normal_matrix = []
        ax = 0
        abnormal_matrix = []
        for row in reader:
            if row[0] != 'teleConfid':
                try:
                    meeting_id = row[1]
                    start_time = row[2]
                    start_time_object = datetime.strptime(start_time[:19], '%Y-%m-%d %H:%M:%S')
                    end_time = row[3]
                    kill_by = row[4]
                    call = float(row[6])
                    if end_time == 'ongoing':
                        end_time_object = datetime.now()
                    else:
                        end_time_object = datetime.strptime(end_time[:19], '%Y-%m-%d %H:%M:%S')
                    time = (end_time_object-start_time_object).days * 24 * 60
                    if time != 0 and call != 0 :
                        ave = (float(call) / float(time))
                        if len(kill_by) > 0:
                            #abnormal_matrix.append([time, call, ave])
                            abnormal_matrix.append([time, ave])
                            ax = ax + 1
                        else:
                            #normal_matrix.append([time, call, ave])
                            normal_matrix.append([time, ave])
                            nx = nx + 1
                except Exception as e:
                    pass
        y = 2
    return normal_matrix, nx, y, abnormal_matrix, ax, y

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
            p[i] = 1 * 10000
            for j in range(0, y):
                num_01 = 1.0/(math.sqrt(2*math.pi) * math.sqrt(sigma[j]))
                num_02 = (-1.0 * math.pow(float(matrix[i][j])-mu[j],2) )/ (2 * sigma[j])
                p[i] = p[i] * num_01 * math.exp(num_02)
        except Exception as pex:
            p[i] = 0
    #print p
    return remove_zero(p)

def remove_zero(p):
    p2 = {}
    y = 0
    for x in p:
        if p[x] != 0:
            p2[y] = p[x]
            y = y +1
    return p2

def ss(p):
    max = p[0]
    min = p[0]
    sum = 0
    for x in p:
        sum = sum + p[x]
        max = p[x] if p[x] > max else max
        min = p[x] if p[x] < min else min
    sigsum = 0
    ave = sum/len(p)
    for x in p:
        sigsum = sigsum + (p[x]-ave)*(p[x]-ave)
    sig = sigsum/len(p)
    return max, min, ave, sig

normal_matrix, nx, y, abnormal_matrix, ax, y = split_data_set('no_press_one.csv')
mu, sigma, x, y = calc_mu_sigma(normal_matrix, nx, y)
print "normal dataset"
print normal_matrix
print "abnormal dataset"
print abnormal_matrix

print "mu, sigma, based on normal data set"
print mu, sigma
p = calc_gaussian_p(normal_matrix, mu, sigma, nx, y)
print "normal data set possibility"
print p
print ss(p)
print "abnormal data set possibility"
p2 = calc_gaussian_p(abnormal_matrix, mu, sigma, ax, y)
print p2
print ss(p2)

sita = 8
true_pos = 0
for x in p2:
    if p2[x] < sita:
        true_pos = true_pos + 1
print "use sita %s to check abnormal dataset" % sita
print true_pos, len(p2), (float(true_pos)/float(len(p2)))*100

false_neg = 0
for x in p:
    if p[x] < sita:
        false_neg = false_neg + 1
print "use sita %s to check normal dataset" % sita
print false_neg, len(p), (float(false_neg)/float(len(p)))*100