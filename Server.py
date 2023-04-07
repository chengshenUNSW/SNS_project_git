import socket
from _thread import *
import threading
import keras
from sklearn.externals import joblib
import pandas as pd
import numpy as np
dirs0 = 'result_mago'
dirs1 = 'result_shamoun'
hypermodel_mago = keras.models.load_model('model_mem60_April_05_0935AM.h5')
animal_data_mago = joblib.load(dirs0 + '/results_data.txt')
queue_input_mago = joblib.load(dirs0 + '/results_queue_.txt')
memory_days_mago = joblib.load(dirs0 + '/memory_days.txt')
pre_days_mago = joblib.load(dirs0 + '/pre_days.txt')

hypermodel_shamoun = joblib.load(dirs1 + '/hypermodel.pkl')
animal_data_shamoun = joblib.load(dirs1 + '/results_data.txt')
queue_input_shamoun = joblib.load(dirs1 + '/results_queue_.txt')
memory_days_shamoun = joblib.load(dirs1 + '/memory_days.txt')
pre_days_shamoun = joblib.load(dirs1 + '/pre_days.txt')

dialogue = {'Can you predict and give me the location of Marbled Godwits or Black-backed Gulls?': \
                'Yes, please give me an actual date after 2012/05/27 and before 2012/07/01 for Marbled Godwits, (Just \
as an example:\"mago:2012/05/28\")\n or an actual date after 2015/05/31 and before 2015/07/01 for Black-backed Gulls. \
(Just as an example:\"shamoun:2015/06/01\")'}

def date_predict_mago(predict_datetime, queue_input_):
    days=len(list(pd.date_range(start=animal_data_mago.index[-1], end=predict_datetime, freq='D')))-1
    if days <=pre_days_mago:
        queue_input_temp=queue_input_
        output_new=hypermodel_mago.predict(queue_input_temp[days-pre_days_mago-1].reshape(1, memory_days_mago, 2))
    else:
        queue_last_output=hypermodel_mago.predict(queue_input_[-pre_days_mago:])
        std_=np.std(animal_data_mago.iloc[:,:-2])
        mean_=np.mean(animal_data_mago.iloc[:,:-2])
        std1=std_
        mean1=mean_
        std2=std_
        mean2=mean_
        for a in range(1,pre_days_mago):
            std1=np.row_stack((std1,std_))
            mean1=np.row_stack((mean1,mean_))
        standardised_last_output=(queue_last_output-mean1)/std1
        for b in range(1,memory_days_mago):
            std2=np.row_stack((std2,std_))
            mean2=np.row_stack((mean2,mean_))
        for c in range(0,pre_days_mago):
            new_input=np.row_stack([queue_input_[-1][1:],[standardised_last_output[c]]])
            queue_input_=np.row_stack([queue_input_,[new_input]])
        for d in range(0,days-pre_days_mago):
            output_temp=hypermodel_mago.predict(queue_input_[-pre_days_mago].reshape(1, memory_days_mago, 2))
            new_input=np.row_stack([queue_input_[-1][1:],[(output_temp[0]-mean_)/std_]])
            queue_input_=np.row_stack([queue_input_,[new_input]])
        output_new=output_temp
    return output_new

def date_predict_shamoun(predict_datetime, queue_input_):
    days=len(list(pd.date_range(start=animal_data_shamoun.index[-1], end=predict_datetime, freq='D')))-1
    if days <=pre_days_shamoun:
        queue_input_temp=queue_input_
        output_new=hypermodel_shamoun.predict(queue_input_temp[days-pre_days_shamoun-1].reshape(1, memory_days_shamoun, 2))
    else:
        queue_last_output=hypermodel_shamoun.predict(queue_input_[-pre_days_shamoun:])
        std_=np.std(animal_data_shamoun.iloc[:,:-2])
        mean_=np.mean(animal_data_shamoun.iloc[:,:-2])
        std1=std_
        mean1=mean_
        std2=std_
        mean2=mean_
        for a in range(1,pre_days_shamoun):
            std1=np.row_stack((std1,std_))
            mean1=np.row_stack((mean1,mean_))
        standardised_last_output=(queue_last_output-mean1)/std1
        for b in range(1,memory_days_shamoun):
            std2=np.row_stack((std2,std_))
            mean2=np.row_stack((mean2,mean_))
        for c in range(0,pre_days_shamoun):
            new_input=np.row_stack([queue_input_[-1][1:],[standardised_last_output[c]]])
            queue_input_=np.row_stack([queue_input_,[new_input]])
        for d in range(0,days-pre_days_shamoun):
            output_temp=hypermodel_shamoun.predict(queue_input_[-pre_days_shamoun].reshape(1, memory_days_shamoun, 2))
            new_input=np.row_stack([queue_input_[-1][1:],[(output_temp[0]-mean_)/std_]])
            queue_input_=np.row_stack([queue_input_,[new_input]])
        output_new=output_temp
    return output_new
print_lock = threading.Lock()
# thread function
def threaded(c):
    while True:
        # data received from client
        data_raw = c.recv(1024)
        data = str(data_raw.decode('ascii'))
        if not data:
            print('Bye')
            # lock released on exit
            print_lock.release()
            break
        else:
            if (data.find('mago')!=-1 and (
                    data.find('2012/06')!=-1 or data.find('2012/05/28')!=-1 or data.find('2012/05/29')!=-1 or \
                    data.find('2012/05/30')!=-1 or data.find('2012/05/31')!=-1)) \
                    or (data.find('2015/06')!=-1 and data.find('shamoun')!=-1):
                type = data.split(':')[0]
                datetime = data.split(':')[1]
                if type == 'mago':
                    predict_date = date_predict_mago(datetime, queue_input_mago)
                if type == 'shamoun':
                    predict_date = date_predict_shamoun(datetime, queue_input_shamoun)
                long = round(100 * predict_date[0][0]) / 100
                lat = round(100 * predict_date[0][1]) / 100
                data = '%f%s%f' % (long, '_', lat)
                print(long)
                print(data)
                # send back location string to client
            elif data in dialogue.keys():
                data = dialogue[data]
            else:
                data = 'You have given me a wrong date or in a wrong format, please check and resend.'
        c.send(data.encode('ascii'))
    # connection closed
    c.close()
def Main():
    host = ""
    port = 12341
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    print("socket binded to post", port)
    # put the socket into listening mode
    s.listen(5)
    print("socket is listening")
    # a forever loop until client wants to exit
    while True:
        # establish connection with client
        c, addr = s.accept()

        # lock acquired by client
        print_lock.acquire()
        print('Connected to :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        start_new_thread(threaded, (c,))
    s.close()
if __name__ == '__main__':
    Main()
