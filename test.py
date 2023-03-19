import json
from datetime import datetime, timedelta
import math
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import scipy.stats
import stat_features
#from d2l import torch as d2l

def read_prompts(path) :
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    output = []
    audio_output = []
    
    for d in data :
        prompts = d['prompts']
        arr = [-1] * 9
        save_stamp = None
        for i in range(len(prompts)) :
            prompt = prompts[i]

            index = prompt['prompt_name'][-1]
            type = prompt['prompt_type']

            if not index.isnumeric():
                if type == 'activity_audio_log' :
                    audio_output.append((stamp, value))
                continue

            index = int(index)
            stamp = prompt['response_stamp']
            
            value = prompt['chosen_response']
            if value != None :
                value = value['response_value']

                if index == 4:
                    value = time_to_val(value)
            else :
                value = -1

            if stamp == None :
                value = -1
            else :
                stamp = stamp[:stamp.index("+")]

                if save_stamp == None :
                    save_stamp = stamp
                    
            arr[index] = int(value)
                
        if save_stamp != None :
            arr[0] = save_stamp
            output.append(arr)
    
    return output, audio_output

def calculate_bearing(lat1, long1, lat2, long2):
    d1 = long2 - long1
    y = math.sin(d1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(d1)
    bearing = math.atan2(y, x)
    bearing = np.rad2deg(bearing)
    return bearing

def calculate_distance(lat1, long1, lat2, long2):
    d1 = (lat2 - lat1) * (lat2 - lat1)
    d2 = (long2 - long1) * (long2 - long1)
    return np.sqrt(d1 + d2)

def close_loc(lat1, long1, lat2, long2):
    """ Determine if one location is within threshold distance to another.
    """
    distance = calculate_distance(lat1, long1, lat2, long2)
    if distance < 0.015:
        return 1
    else:
        return 0


def calculate_average(data) :
    sum = 0
    for i in range(0,len(data)) :
        sum += data[i]
    return sum / len(data)

def calculate_stddev(data, current) :
    stddev = 0
    for i in range(0, len(data)) :
        stddev = stddev + pow(data[i]-current, 2)
       
    return pow(stddev, 0.5)

def find_max_min(data) :
    min = 0
    max = 0
    for i in range(0, len(data)) :
        if data[i] > max :
            max = data[i]
        elif data[i] < min :
            min = data[i]

    return max, min

def read_data_better(path, responses, interactions, read_range, home_coords) :
    file = open(path, 'r')

    file.readline()
    st = None

    read_index = 0
    data = []
    all_data = []
    heartrate = 70
    rotation = 0
    acceleration = 0
    lat = home_coords[0]
    long = home_coords[1]

    (xr, yr, zr) = (0,0,0)
    (xa, ya, za) = (0,0,0)

    last_response = 0
    last_time = None

    # get 30 minutes of data
    # get averages from this period
    # 

    while st != '' and read_index < read_range:
        st = file.readline()

        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == None :
            if 'heart' in st :
                heartrate = fields['sensors']['heart_rate']
                continue
        
        if fields['message_type'] == 'location' :
            sensor = fields['sensors']
            lat = float(sensor['latitude'])
            long = float(sensor['longitude'])
            lat = round(lat, 2)
            long = round(long, 2)
            continue     

        stamp = str(fields["stamp"])  
        stamp = stamp[:stamp.index("+")]

        time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        if last_time == None :
            last_time = time

        response_index, time_left = in_interval(time, responses, last_response)

        sensor = fields["sensors"]

        if response_index != last_response :
            cl = close_loc(lat, long, home_coords[0], home_coords[1])
            all_data.append(analyze_data(data,interactions[last_response], last_response, cl))
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)
            last_time = time
            data = []
            last_response = response_index

        #gets current rotation
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        #gets current rotation
        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']
        
        if (time-last_time).total_seconds() >= 1 :
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)
            last_time = time
            #data.append([xr, yr, zr, xa, ya, za, heartrate, close_loc(lat, long, home_coords[0], home_coords[1])])
            data.append([rotation, acceleration, heartrate])
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)

            read_index += 1
            print(read_index)
    if response_index == last_response :
        cl = close_loc(lat, long, home_coords[0], home_coords[1])
        all_data.append(analyze_data(data,interactions[last_response], last_response, cl))
    return all_data
           
def analyze_data(data, interaction, index, cl) :
    new_data = []
    data = np.array(data)
    r = data[:,0]
    a = data[:,1]
    h = data[:,2]
    interaction = np.array(interaction)
    s1 = stat_features.generate_statistical_features(r)
    s2 = stat_features.generate_statistical_features(a)
    s3 = stat_features.generate_statistical_features(h)
    s4 = stat_features.generate_statistical_features(interaction)
    new_data += ([s1] + [s2] + [s3] + [s4] + [index] )
    #new_data += (s1 + s2 + s3 + s4 + [cl] + [index] )
    # r_a = np.mean(r)
    # a_a = np.mean(a)
    # h_a = np.mean(h)

    # r_d = np.std(r)
    # a_d = np.std(a)
    # h_d = np.std(h)
    # h_max = np.max(h)
    # h_min = np.min(h)

    
    
    # i_sum = 0
    # i_avg = 0
    # i_max = 0
    # i_min = 0
    # i_std = 0
    # if interaction.size != 0 :
    #     i_sum = np.sum(interaction)
    #     i_avg = np.average(interaction)
    #     i_max = np.max(interaction)
    #     i_min = np.min(interaction)
    #     i_std = np.std(interaction)

    # for i in range(0, len(data)-60, 60) :
    #     r1_a = np.mean(r[i:i+60])
    #     a1_a = np.mean(a[i:i+60])
    #     h1_a = np.mean(h[i:i+60])

    #     r1_d = np.std(r[i:i+60])
    #     a1_d = np.std(a[i:i+60])
    #     h1_d = np.std(h[i:i+60])
    #     #r1_a, a1_a, h1_a, r1_d, a1_d, h1_d, 
    #     new_data.append([r1_a, a1_a, h1_a, r1_d, a1_d, h1_d, r_a, a_a, h_a, r_d, a_d, h_d, h_max, h_min, i_sum, i_avg, i_max, i_min, i_std, interaction.size, index])
    return new_data
        



def read_data(path, responses, interactions, read_range, home_coords) :
    file = open(path, 'r')

    file.readline()
    st = None

    read_index = 0
    index = 0
    data = []
    heartrate = 70
    rotation = 0
    acceleration = 0

    average_heartrate = [heartrate] * 20
    stddev_heartrate = [heartrate] * 5
    average_accel = [0] * 20
    average_rotation = [0] * 20

    lat = home_coords[0]
    long = home_coords[1]

    (xr, yr, zr) = (0,0,0)
    (xa, ya, za) = (0,0,0)
    reset = True

    while st != '' and read_index < read_range:
        st = file.readline()

        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == None :
            if 'heart' in st :
                heartrate = fields['sensors']['heart_rate']
                continue
        
        if fields['message_type'] == 'location' :
            sensor = fields['sensors']
            lat = float(sensor['latitude'])
            long = float(sensor['longitude'])
            lat = round(lat, 2)
            long = round(long, 2)
            continue

        stamp = str(fields["stamp"])  
        stamp = stamp[:stamp.index("+")]

        time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")

        response_index, time_left = in_interval(time, responses)

        sensor = fields["sensors"] 

        if reset :
            average_heartrate = [heartrate] * 20
            stddev_heartrate = [heartrate] * 5
            average_accel = [0] * 20
            average_rotation = [0] * 20
            index = 0
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)
            reset = False 

        #gets current rotation
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        #gets current rotation
        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']

        #sensor runs at 10 hz so save the data every second
        if (index != 0 and index % 10 == 0) :
            #gets size of vectors
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)

            average_heartrate[index % len(average_heartrate)] = heartrate
            stddev_heartrate[index % len(stddev_heartrate)] = heartrate
            
            average_rotation[index % len(average_rotation)] = rotation
            average_accel[index % len(average_accel)] = acceleration

            avg_h = calculate_average(average_heartrate)
            std_h = calculate_stddev(stddev_heartrate, avg_h)
            avg_r = calculate_average(average_rotation)
            avg_a = calculate_average(average_accel)
            max, min = find_max_min(average_heartrate)
            a_max, a_min = find_max_min(average_rotation)

            last_index = response_index - 1 if response_index - 1 > 0 else 0
            last_responses = responses[last_index]

            arr = [rotation, acceleration, heartrate,avg_h,std_h,avg_r,avg_a,max, min, interactions[response_index], close_loc(lat, long, home_coords[0], home_coords[1])]

            # for val in last_responses[1:] :
            #     arr.append(int(val))

            arr.append(response_index)

            data.append(arr)

            #resets vectors
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)
            read_index += 1
        index += 1
        
    file.close()
    return data

def time_to_val(time) :
    if time == 'None' :
        return 0
    elif 'Less' in time :
        return 1
    else :
        return 2

def find_home(path, read_range) :
    file = open(path, 'r')

    st = None

    freq = dict()
    file.readline()
    index = 0
    while st != '' and index < read_range * 5 :
        st = file.readline()

        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == 'location' :
            sensor = fields['sensors']
            lat = float(sensor['latitude'])
            long = float(sensor['longitude'])
            lat = round(lat, 2)
            long = round(long, 2)

            if freq.get((lat,long)) == None :
                freq[(lat,long)] = 1
            else :
                freq[(lat,long)] += 1
        index += 1
    file.close()

    freq = sorted(freq.items(), key=lambda x:x[1], reverse=True)
    return freq, freq[0][0]


def in_interval(stamp, responses, last_index) :
    index = last_index

    while index < len(responses) :
        response = responses[index]
        time = datetime.strptime(response[0], "%Y-%m-%dT%H:%M:%S.%f")
        diff = time - stamp
        diff = diff.total_seconds()

        if diff >= 0 :
            if diff <= 1800 :
                return index, 0
            else :
                return -1, diff
        index += 1
    return -1, 0

def get_y(responses, data, y_index) :
    return [int(responses[data[i][-1]][y_index]) for i in range(0,len(data))]

def train_model(responses, data, y_index) :
    npdata = np.array(data)
    X = npdata[:,0:len(data[0]) - 1]
    y = np.array(get_y(responses, data, y_index))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
    #oversample = SMOTE()
    #X_train, y_train = oversample.fit_resample(X_train, y_train)
    clf = MLPClassifier(random_state=1, max_iter=3000,activation='tanh',hidden_layer_sizes=(120,60)).fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    #test_accuracy(clf,X_test[0:2000],y_test[0:2000])
    return clf

def train(X_train, X_test, y_train, y_test) :
    #oversample = SMOTE()
    #X_train, y_train = oversample.fit_resample(X_train, y_train)
    clf = MLPClassifier(random_state=1, max_iter=300000,activation='tanh',hidden_layer_sizes=(12,6),epsilon=1e-12).fit(X_train, y_train)
    print(clf.score(X_test,y_test))
    #test_accuracy(clf,X_test[0:2000],y_test[0:2000])
    return clf, clf.score(X_test, y_test)

# tests model accuracy
def test_accuracy(model, X, y) :
    predictions = model.predict(X)
    correct_outputs = y
    tn, fp, fn, tp = metrics.confusion_matrix(correct_outputs, predictions).ravel()
    print("Accuracy = %f" % (metrics.accuracy_score(correct_outputs, predictions)))
    print("TN = %d FP = %d FN = %d TP = %d" % (tn, fp, fn, tp))
    print(metrics.classification_report(correct_outputs, predictions, target_names = ["Wrong", "Right"]))

def plot_data(data, responses) :
    #heartrate = 2
    

    x = [i for i in range(len(data))]
    y1 = get_y(responses, data, 1)
    y2 = [data[i][2] for i in range(0,len(data))]
    fig, ax = plt.subplots()
    ax.plot(x, y1, "r--")
    ax.plot(x, y2, "b")
    plt.xlabel("Heartrate")
    plt.ylabel("Feeling")
    plt.title("Graph")
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 30))
    plt.show()

def get_interactions(log_path, responses) :
    file = open(log_path, 'r')

    st = file.readline()

    interaction_durations = [None] * len(responses)
    for i in range(len(interaction_durations)) :
        interaction_durations[i] = []

    while st :
        if "Proximity interaction ended:" in st :
            time = file.readline()
            duration = file.readline()
            time = time[time.index(' -') + 3:-1]
            duration = float(duration[duration.index('=') + 2:-2])

            if duration < 100 :
                st = file.readline()
                continue
            
            if '.' in time :
                time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
            else :
                time = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            time += timedelta(hours=4)
            if time.month > 11 or (time.month == 11 and time.day >= 5) :
                time += timedelta(hours=1)
            index, time_left = in_interval(time, responses, 0)

            if index != -1 :
                interaction_durations[index].append(duration)
                

        st = file.readline()

    return interaction_durations

def get_data(range) :
    dyad = "dyads/dyadH05A2w"
    path = dyad + ".prompt_groups.json"
    data_path = "clean_sensor_data.json"
    log_path = dyad + ".system_logs.log"
    freqs, home_cords = find_home("clean_sensor_data.json", 5000)
    responses, audio_responses = read_prompts(path)
    for response in responses :
        print('response: ' + str(response[1]) + ' ' + str(response[2]) + ' ' + str(response[3]) + ' ' + str(response[2]))
    interactions = get_interactions(log_path, responses)
    data = read_data_better(data_path, responses, interactions, range, home_cords)

    return data

def run() :
    dyad = "h01/dyadH01A1w"
    path = dyad + ".prompt_groups.json"
    data_path = dyad + ".sensor_data.json"
    log_path = dyad + ".system_logs.log"
    read_range = 256000
    freqs, home_cords = find_home(data_path, 10000)
    responses, audio_responses = read_prompts(path)

   

    interactions = get_interactions(log_path, responses)
    data = read_data(data_path, responses, interactions, read_range, home_cords)
    #plot_data(data, responses)
    for i in range(1, len(responses[0])) :
        clf = train_model(responses, data, i)
    x = 10

if __name__ == '__main__' :
    get_data(10000)