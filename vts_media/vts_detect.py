from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
import time

#"CALL_SSQ","LOG_SOURCE","POP","LOG_HOST","WBXMEETINGID","WBXCONFID","TRANSMIT_TYPE","AUDIO_LOSS","AUDIO_LOSS_PERCENT","MVIDEO_LOSS",
# "MVIDEO_LOSS_PERCENT","EVIDEO_LOSS","EVIDEO_LOSS_PERCENT","EVENT_TIME","LEG_TYPE","CALLEE","CALLER","RAW_MESSAGE"

sequence_length = 100

class VTS_DATA():
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

    def __init__(self, line, id):
        row = line.split(",")
        self.id = id
        self.call_ssq = row[0]
        self.leg_type = row[14]
        self.callee = row[15]
        self.caller = row[16]
        self.audio_loss_percent = float(row[8])
        self.mvideo_loss_percent = float(row[10])
        self.evideo_loss_percent = float(row[12])


def calculate_mean_error(dataset):
    size = len(dataset)
    print "size="+str(size)
    total = 0.0
    max = 0
    min = 99999999
    non_zero_size = 0
    for vts_data in dataset:
        total = total + vts_data.audio_loss_percent
        if vts_data.audio_loss_percent > max:
            max = vts_data.audio_loss_percent
        if vts_data.audio_loss_percent < min:
            min = vts_data.audio_loss_percent
        if vts_data.audio_loss_percent > 0:
            non_zero_size = non_zero_size + 1
    print "max = %s, min = %s" % (max , min)
    print "non zero size= %s" % (non_zero_size)
    mean = (float)(total) / (float)(non_zero_size)
    sq_total = 0.0
    for vts_data in dataset:
        if vts_data.audio_loss_percent > 0:
            sq_total = sq_total + (vts_data.audio_loss_percent - mean)*(vts_data.audio_loss_percent - mean)
    sqe = sq_total / mean
    print "mean = %s, sqe = %s" % (mean, sqe)
    return mean, sqe, non_zero_size

def processData(file="avw_vts_mstat_raw.csv"):
    #raw_dataset = numpy.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, dtype="string")
    #raw_dataset = numpy.genfromtxt(file, delimiter=',')
    with open(file) as f:
        lines = f.readlines()

    dataset = []
    index = 0
    for line in lines:
        index = index + 1
        if index == 1:
            continue
        vtsdata = VTS_DATA(line, index)
        dataset.append(vtsdata)

    return dataset

def drawPlot(dataset):
    plot_x = []
    plot_y = []

    for vts_data in dataset:
        plot_x.append(vts_data.id)
        plot_y.append(vts_data.audio_loss_percent)

    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    ax1.set_title("audio_loss_percent")
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(plot_x, plot_y, c='b', marker='o')
    plt.legend('x1')

    #plt.show()
    fig.savefig("VTS_audio_loss_percent.png")
    fig.clear()


#Use gaussian model to determine anomaly point, set 0 or 1
#Use LSTM to detect time series anomaly
#Alert for LSTM output, calculate related Caller/Callee, meeting , site, send out report

def Gaussian(dataset):
    mean, sqe, non_zero_size = calculate_mean_error(dataset)
    threshold = 0.00024914
    total_ano = 0
    max_p = 0
    min_p = 99999999
    total_p = 0
    size = len(dataset)
    for vts_data in dataset:
        if vts_data.audio_loss_percent > 0:
            num_01 = 1.0 / (math.sqrt(2 * math.pi) * math.sqrt(sqe))
            num_02 = (-1.0 * math.pow(float(vts_data.audio_loss_percent - mean), 2) / (2 * sqe))
            p = 1 * num_01 * math.exp(num_02)
            vts_data.gaussian_possibility = p
            if p > max_p:
                max_p = p
            if p < min_p:
                min_p = p
            total_p = total_p + p
            if p < threshold:
                vts_data.anomaly = 1
                total_ano = total_ano + 1
    mean_p = total_p / non_zero_size
    print "p: max=%s,min=%s,mean=%s,total_ano=%s" % (max_p, min_p, mean_p, total_ano)
    return dataset

def build_model():
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
            input_length=sequence_length - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model


def main():
    dataset= processData()
    dataset = Gaussian(dataset)
    #drawPlot(dataset[0], 'Audio')
    #drawPlot(poss, 'Poss')
    #drawPlot(dataset[1], 'MVudio')
    #drawPlot(dataset[2], 'EVudio')
    nn_model = build_model()
    nn_model.train_on_batch()

if __name__ == "__main__":
    main()