import numpy
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import math

def processData(file="no_press_one.csv"):
    raw_dataset = numpy.loadtxt(open(file, "rb"), delimiter=",", skiprows=1, dtype="string")
    dataset = []
    resultset = []
    plot_x_1 = []
    plot_y_1 = []
    plot_x_2 = []
    plot_y_2 = []
    for row in raw_dataset:
        try:
            #print row
            start_time = row[2][1:20]
            start_time_object = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            end_time = row[3][1:20]
            kill_by = row[4]
            call = math.log(float(row[6]), math.e)
            if 'ongoing' in end_time:
                end_time_object = datetime.strptime('2016-04-10 23:59:59', '%Y-%m-%d %H:%M:%S')
            else:
                end_time_object = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
            time = (end_time_object - start_time_object).days * 24 * 60
            time = math.log(time, math.e)
            kill = 0
            if kill_by != '' and len(kill_by) > 0:
                kill = 1

            if time < 8:
                kill = 1

            if time > 0 and call > 0:
                new_row = [time, call]
                print new_row, kill
                dataset.append(new_row)
                resultset.append([kill])
                if kill == 0:
                    plot_x_1.append(time)
                    plot_y_1.append(call)
                else:
                    plot_x_2.append(time)
                    plot_y_2.append(call)

        except:
            pass
    return dataset, resultset, plot_x_1, plot_y_1, plot_x_2, plot_y_2

def drawPlot(plot_x_1, plot_y_1, plot_x_2, plot_y_2):

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('TeleFraud')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(plot_x_1, plot_y_1, c='b', marker='o')
    plt.legend('x1')

    ax2 = fig.add_subplot(111)
    ax2.scatter(plot_x_2, plot_y_2, c='r', marker='x')

    #plt.show()
    fig.savefig("teleFraud_plot_ln.png")


def NN(dataset, resultset):

    # Create the neural network model
    model = Sequential([
        Dense(4, input_dim=2), Activation('sigmoid'),
        Dense(4, input_dim=4), Activation('sigmoid'),
        Dense(1), Activation('sigmoid')
    ])

    # Activation: sigmoid, relu, tanh, softmax
    # optimizer: SGD, RMSprop
    # loss: mean_squared_error, binary_crossentropy

    # Compile the model with an optimiser
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.6, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy')

    # Create training cases for an XOR function
    #model.load_weights("telefraud.hdf")
    model.fit(dataset, resultset, epochs=400)
    print model.get_weights()
    model.save("telefraud.hdf")
    predict_result =  model.predict(dataset, batch_size=320, verbose=1)
    for x in predict_result:
        print x
    #progress_plot(model, x, y, bottomLeft, topRight, epochList)



def main():
    dataset, resultset, plot_x_1, plot_y_1, plot_x_2, plot_y_2 = processData()
    #drawPlot(plot_x_1, plot_y_1, plot_x_2, plot_y_2)
    NN(dataset, resultset)

if __name__ == "__main__":
    main()