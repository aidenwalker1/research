import json
from datetime import datetime

def read_prompts(path) :
    file = open(path, 'r')
    data = json.load(file)
    file.close()

    output = []
    audio_output = []
    
    for d in data :
        prompts = d['prompts']
        arr = [0] * 8
        save_stamp = None
        print(len(prompts))
        for i in range(len(prompts)) :
            prompt = prompts[i]

            index = prompt['prompt_name'][-1]
            type = prompt['prompt_type']

            if not index.isnumeric() or int(index) == 8:
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

def clean_data(path, responses) :
    file = open(path, 'r')
    cleaned_data = open('clean_sensor_data.json', 'w')
    st = file.readline()

    while st != '':
        st = file.readline()

        fields = json.loads(st[:st.find(',\n')])

        if fields['message_type'] == None :
            if 'heart' in st :
                cleaned_data.write(st)
                continue

        if fields['message_type'] == 'location' :
            cleaned_data.write(st)
            continue

        if fields['message_type'] != 'device_motion' :
            continue

        stamp = str(fields["stamp"])
        stamp = stamp[:stamp.index("+")]
        time = None
        if '.' in stamp :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f")
        else :
            time = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
            index = st.index("+")
            st = st[:index] + ".0" + st[index:]

        response_index = in_interval(time, responses)

        if response_index == -1:
            continue

        cleaned_data.write(st)
    file.close()
    cleaned_data.close()

def time_to_val(time) :
    if time == 'None' :
        return 0
    elif 'Less' in time :
        return 1
    else :
        return 2
    
def in_interval(stamp, responses) :
    index = 0
    time = None

    while index < len(responses) :
        response = responses[index]
        if '.' in response[0] :
            time = datetime.strptime(response[0], "%Y-%m-%dT%H:%M:%S.%f")
        else :
            time = datetime.strptime(response[0], "%Y-%m-%dT%H:%M:%S")
        diff = time - stamp
        diff = diff.total_seconds()

        if diff >= 0 :
            if diff <= 1800 :
                return index
            else :
                return -1
        index += 1
    return -1

dyad = "dyads/dyadH05A1w"
path = dyad + ".prompt_groups.json"
data_path = dyad + ".sensor_data.json"
log_path = dyad + ".system_logs.log"
responses, audio_responses = read_prompts(path)
clean_data(data_path, responses)