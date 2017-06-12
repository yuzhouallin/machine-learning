import numpy
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import math

#"CALL_SSQ","LOG_SOURCE","POP","LOG_HOST","WBXMEETINGID","WBXCONFID","TRANSMIT_TYPE","AUDIO_LOSS","AUDIO_LOSS_PERCENT","MVIDEO_LOSS",
# "MVIDEO_LOSS_PERCENT","EVIDEO_LOSS","EVIDEO_LOSS_PERCENT","EVENT_TIME","LEG_TYPE","CALLEE","CALLER","RAW_MESSAGE"

class vts_data():
    id = 0
    call_ssq = ''
    leg_type = ''
    callee = ''
    caller = ''
    audio_loss_percent = 0.0
    mvideo_loss_percent = 0.0
    evideo_loss_percent = 0.0
    anomaly = 0
    gaussian_possibility = 0.0



def remove_zero(dataset):
    new_ds = []
    for data in dataset:
        if data[1] > 0:
            new_ds.append([data[0], data[1]])
    return new_ds

def calculate_mean_error(dataset):
    size = len(dataset)
    print "size="+str(size)
    total = 0.0
    max = 0
    min = 99999999
    for data in dataset:
        total = total + data[1]
        if data[1] > max:
            max = data[1]
        if data[1] < min:
            min = data[1]
    print "max = %s, min = %s" % (max , min)
    mean = (float)(total) / (float)(size)
    sq_total = 0.0
    for data in dataset:
        sq_total = sq_total + (float(data[1]) - mean)*(data[1] - mean)
    sqe = sq_total / mean
    print "mean = %s, sqe = %s" % (mean, sqe)
    return mean, sqe

def processData(file="avw_vts_mstat_raw.csv"):
    #raw_dataset = numpy.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, dtype="string")
    #raw_dataset = numpy.genfromtxt(file, delimiter=',')
    with open(file) as f:
        lines = f.readlines()

    dataset = [[],[],[]]
    index = 0
    for line in lines:
        index = index + 1
        if index == 1:
            continue
        try:
            row = line.split(",")
            audio_loss = float(row[8])
            mvideo_loss = float(row[10])
            evideo_loss = float(row[12])
            dataset[0].append([index, audio_loss])
            dataset[1].append([index, mvideo_loss])
            dataset[2].append([index, evideo_loss])
        except:
            pass
    return dataset

def drawPlot(dataset, title):
    plot_x = []
    plot_y = []

    for data in dataset:
        plot_x.append(data[0])
        plot_y.append(data[1])

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(plot_x, plot_y, c='b', marker='o')
    plt.legend('x1')

    #plt.show()
    fig.savefig("VTS_"+title+".png")
    fig.clear()


#Use gaussian model to determine anomaly point, set 0 or 1
#Use LSTM to detect time series anomaly
#Alert for LSTM output, calculate related Caller/Callee, meeting , site, send out report

def Gaussian(dataset):
    mean, sqe = calculate_mean_error(dataset)
    pset = []
    threshold = 0.00024914
    total_ano = 0
    ano_indexs = []
    max_p = 0
    min_p = 99999999
    total_p = 0
    size = len(dataset)
    for data in dataset:
        num_01 = 1.0 / (math.sqrt(2 * math.pi) * math.sqrt(sqe))
        num_02 = (-1.0 * math.pow(float(data[1] - mean), 2) / (2 * sqe))
        p = 1 * num_01 * math.exp(num_02)
        pset.append([data[0],p])
        if p > max_p:
            max_p = p
        if p < min_p:
            min_p = p
        total_p = total_p + p
        if p < threshold:
            total_ano = total_ano + 1
            ano_indexs.append(data[0])
            print data[1]
    mean_p = total_p / size
    print "p: max=%s,min=%s,mean=%s,total_ano=%s" % (max_p, min_p, mean_p, total_ano)
    return pset

def NN(dataset):
    a = 1




def main():
    dataset= processData()
    poss = Gaussian(remove_zero(dataset[0]))
    #drawPlot(dataset[0], 'Audio')
    #drawPlot(poss, 'Poss')
    #drawPlot(dataset[1], 'MVudio')
    #drawPlot(dataset[2], 'EVudio')
    NN(dataset)

if __name__ == "__main__":
    main()