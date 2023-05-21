import os
import json
from tqdm import tqdm


def get_files_in_folder(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]


def count_api(api_dict, file_path):
    data = json.load(open(file_path))['data']
    list_api_call = []

    for method in data:
        for api in method["api"]:
            in_dict = api["full_api_call"] in api_dict
            in_list_api = api["full_api_call"] in list_api_call
            
            if in_dict and in_list_api:
                api_dict[api['full_api_call']]['call_num'] += 1
            
            elif in_dict and not in_list_api:
                list_api_call.append(api['full_api_call'])
                api_dict[api['full_api_call']]['call_num'] += 1
                api_dict[api['full_api_call']]['app_num'] += 1
            
            elif not in_dict and not in_list_api:
                list_api_call.append(api['full_api_call'])
                api_dict[api['full_api_call']] = {
                    'call_num': 1,
                    'app_num': 1
                }
    
    return api_dict


dataset_training_path = '../Dataset/Training'
folder_names = ['Adware_600', 'Banking_600', 'Benign_600', 'Riskware_600', 'SMS_600']

for folder_name in folder_names:
    print(f'Processing {folder_name}')
    json_files = get_files_in_folder(f'{dataset_training_path}/{folder_name}')
    api_dict = {}
    
    for json_file in tqdm(json_files):
        api_dict = count_api(api_dict, f'{dataset_training_path}/{folder_name}/{json_file}')

    open(f"./output/All_APIs_in_{folder_name}.json", "w").write(json.dumps(api_dict, indent=4))
