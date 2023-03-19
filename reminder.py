import json
from datetime import datetime
import numpy as np
import math

import matplotlib.pyplot as plt
from test import read_prompts, in_interval

num_features = 2

#reads sensor data from sensor.json
def read_sensor_data(filename, seconds, responses) -> np.array:
    file = open(filename, 'r')

    file.readline()

    i = 0
    st = ''
    fields = []

    data = []
    
    (xr, yr, zr) = (0,0,0)
    (xa, ya, za) = (0,0,0)

    current_second = 0

    last_time = None
    time = None
    last_index = 0
    #reads through 
    while current_second < seconds:
        st = file.readline()

        #if reach end of json
        if ']' in st :
            break

        fields = json.loads(st[:st.find(',\n')])
        stamp = str(fields["stamp"])  
        stamp = stamp[:stamp.index("+")]
        if '.' in stamp :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        else :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
        
        dif = 1

        if time.hour > 20 or (time.hour == 20 and time.minute > 15) :
            last_time = time
            continue
        elif time.hour < 8 :
            last_time = time
            continue
        #makes sure current line is watch device motion
        if fields['message_type'] != 'device_motion' :
            continue
        
        dif = 1

        sensor = fields["sensors"] 
        
        #gets current rotation
        xr += sensor['rotation_rate_x']
        yr += sensor['rotation_rate_y']
        zr += sensor['rotation_rate_z']

        #gets current rotation
        xa += sensor['user_acceleration_x']
        ya += sensor['user_acceleration_y']
        za += sensor['user_acceleration_z']
        
        #sensor runs at 10 hz so save the data every second
        if (i != 0 and i % 10 == 0) :
            #gets size of vectors
            rotation = math.pow(xr**2 + yr**2 + zr**2, 1/2)
            acceleration = math.pow(xa**2 + ya**2 + za**2, 1/2)

            
            if last_time != None :
                dif = (time - last_time).total_seconds()
                if dif >= 2 :
                    for i in range(math.ceil(dif)) :
                        data.append([0,0, time])
            last_time = time
            #appends last second of data
            data.append([rotation, acceleration, time])

            #resets vectors
            (xr, yr, zr) = (0,0,0)
            (xa, ya, za) = (0,0,0)

            current_second += dif
            print(current_second)

        i+=1
    file.close()
    return np.array(data)

def find_gaps(filename, seconds) -> list:
    file = open(filename, 'r')

    file.readline()
    st = ''
    fields = None

    current_second = 0
    last_time = None
    skipped_times = []
    i = 0
    while current_second < seconds:
        st = file.readline()

        #if reach end of json
        if ']' in st :
            break
        fields = json.loads(st[:st.find(',\n')])

        #makes sure current line is watch device motion
        if fields['message_type'] != 'device_motion':
            continue

        if i != 0 and i % 10 == 0 :

            stamp = str(fields["stamp"])  
            stamp = stamp[:stamp.index("+")]

            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")

            if last_time != None :
                dif = (time - last_time).total_seconds()
                if dif > 3 :
                    print(str(dif))
                    skipped_times.append([last_time, time])

            last_time = time
            current_second += 1
        i += 1

    file.close()
    return skipped_times


#creates list of rotation and acceleration
def fill_features(rotation, acceleration) -> list:
    attribute_list = [None] * (num_features)

    attribute_list[0] = rotation
    attribute_list[1] = acceleration

    return attribute_list

#plots acceleration + rotation data
def plot_data(data, lower, upper) :
    #gets acceleration and rotation
    y1 = [data[i][1] for i in range(lower, upper)]
    y2 = [data[i][0] for i in range(lower, upper)]
    x = [i for i in range(lower, upper)]

    fig, ax = plt.subplots()

    #plots data
    ax.plot(x, y1, "red")
    ax.plot(x, y2, "b--")

    #labels
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration/Rotation")
    plt.title("Acceleration/Rotation over time")

    #x ticks
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 1000))
    #shows data
    plt.show()

#alerts the user if total seconds not moving above threshold
def alert_user(data) -> bool:
    total = 0

    #combines acceleration + rotation
    for i in range(len(data)) :
        if data[i][0] < 0.025 :
            total += 1
    
    print("total: " + str(total))

    #needs less than 3 minutes of movement in the hour to notify
    if total > 3420 :
        print("YOU NEED TO WEAR THE WATCH")
        return True
    return False

def was_watch_worn(data) -> bool :
    total = 0
    for i in range(0,300) :
        if data[i][0] < 0.025 :
            total += 1
    return total < 295

def find_when_watch_worn(data) -> list:
    times = []
    total = len(data)
    not_worn = 0
    for i in range(0, len(data) - 300, 300) :
        b = was_watch_worn(data[i:i+300,:])
        if not b :
            not_worn += 300
        times.append(b)
    return times, ((total-not_worn)/total)

def plot_time(day_times) :
    length = len(day_times)
    #gets acceleration and rotation
    y2 = [1 if i else 0 for i in day_times]
    x = [i for i in range(length)]

    fig, ax = plt.subplots()

    #plots data
    ax.plot(x, y2, "b")
    #labels
    plt.xlabel("Time (5's of minutes)")
    plt.ylabel("Watch was worn")
    plt.title("When watch was worn")

    #x ticks
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 5))
    #shows data
    plt.show()
def plot_times(night_times, day_times) :
    min_len = min(len(night_times), len(day_times))
    night_times = night_times[:min_len]
    day_times = day_times[:min_len]
    #gets acceleration and rotation
    y1 = [1 if i else 0 for i in night_times]
    y2 = [1 if i else 0 for i in day_times]
    x = [i for i in range(min_len)]

    fig, ax = plt.subplots()

    #plots data
    ax.plot(x, y1, "red")
    ax.plot(x, y2, "b--")
    #labels
    plt.xlabel("Time (5's of minutes)")
    plt.ylabel("Watch was worn")
    plt.title("When watch was worn")

    #x ticks
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 5))
    #shows data
    plt.show()

def run_program() :
    dyad = "dyads/dyadH05A1w"
    path = dyad + ".prompt_groups.json"
    data_path = ".sensor_data.json"
    log_path = dyad + ".system_logs.log"
    read_range = 256000 # in seconds

    #skipped_times = find_gaps(day_path,read_range)
    #print(str(len(skipped_times)))
    #night_data = read_sensor_data(night_path, read_range)
    responses, _ = read_prompts(path)
    day_data = read_sensor_data(dyad + data_path, read_range, responses)

    #night_times = find_when_watch_worn(night_data)
    day_times, percentage = find_when_watch_worn(day_data)
    print(percentage)
    #plot_times(night_times, day_times)
   # min_len = min(len(day_data), len(night_data))
    plot_time(day_times)
    #plot_data(day_data, 0, len(day_data))
    #plot_data(night_data, 0, min_len)
    x = 10

if __name__ == '__main__' :
    run_program()