import os
import json
import pandas as pd
import numpy as np
from functools import reduce
from path import Path


# adware_dataset_path = '../Dataset/Training/Adware_600'
# banking_dataset_path = '../Dataset/Training/Banking_600'
# benign_dataset_path = '../Dataset/Training/Benign_600'
# riskware_dataset_path = '../Dataset/Training/Riskware_600'
# sms_dataset_path = '../Dataset/Training/SMS_600'

adware_dataset_path = '../Dataset/Testing/Adware_800'
banking_dataset_path = '../Dataset/Testing/Banking_800'
benign_dataset_path = '../Dataset/Testing/Benign_800'
riskware_dataset_path = '../Dataset/Testing/Riskware_800'
sms_dataset_path = '../Dataset/Testing/SMS_800'

top_apis_path = '../CountAPIs/output/Sorted_APIs_in_All.json'

num_top = 400


def list_files(path: str):
    return [f'{path}/{f}' for f in os.listdir(path)]


def create_app_api(save_path: str):
    api_dataset = json.load(open(top_apis_path, "r"))[:num_top]
    api_names = [api['api_name'] for api in api_dataset]

    def create_row(file: str, label: str):
        content = json.load(open(file, "r"))["data"]
        result = np.zeros((len(api_names)), dtype=int)

        for method in content:
            for api_call in method["api"]:
                if api_call["full_api_call"] in api_names:
                    result[api_names.index(api_call["full_api_call"])] = 1
        result = result.tolist()
        result.append(label)
        return result
    
    result = []
    result.append(list(map(lambda x: create_row(x, "Adware"), list_files(adware_dataset_path))))
    result.append(list(map(lambda x: create_row(x, "Banking"), list_files(banking_dataset_path))))
    result.append(list(map(lambda x: create_row(x, "Benign"), list_files(benign_dataset_path))))
    result.append(list(map(lambda x: create_row(x, "Riskware"), list_files(riskware_dataset_path))))
    result.append(list(map(lambda x: create_row(x, "SMS"), list_files(sms_dataset_path))))
    matrix = list(reduce(lambda x, y: np.concatenate((x, y)), result))

    api_names.append("Label")

    pd.DataFrame(matrix, columns=api_names).to_csv(save_path)
    return matrix


def save_int_csv(path, obj):
    np.savetxt(path, obj, fmt='%d', delimiter=',')


def create_invoke(save_path: str):
    api_dataset = json.load(open(top_apis_path, "r"))[:num_top]
    api_names = [api['api_name'] for api in api_dataset]
    invoke_matrix = np.zeros((len(api_names), len(api_names)), dtype=np.int32)

    def process(app):
        invoke_static = set()
        invoke_virtual = set()
        invoke_direct = set()
        invoke_super = set()
        invoke_interface = set()
        data = json.load(open(app, "r"))["data"]
        for method in data:
            apis = method["api"]
            for api in apis:
                if api["full_api_call"] in api_names:
                    invoke = api["invoke"]
                    if invoke == 'invoke-static':
                        invoke_static.add(api_names.index(api["full_api_call"]))
                    elif invoke == 'invoke-virtual':
                        invoke_virtual.add(api_names.index(api["full_api_call"]))
                    elif invoke == 'invoke-direct':
                        invoke_direct.add(api_names.index(api["full_api_call"]))
                    elif invoke == 'invoke-super':
                        invoke_super.add(api_names.index(api["full_api_call"]))
                    elif invoke == 'invoke-interface':
                        invoke_interface.add(api_names.index(api["full_api_call"]))
        all_type = []
        all_type.append(invoke_static)
        all_type.append(invoke_virtual)
        all_type.append(invoke_direct)
        all_type.append(invoke_super)
        all_type.append(invoke_interface)

        return all_type

    result = []
    result = np.concatenate((result, list_files(adware_dataset_path)))
    result = np.concatenate((result, list_files(banking_dataset_path)))
    result = np.concatenate((result, list_files(benign_dataset_path)))
    result = np.concatenate((result, list_files(riskware_dataset_path)))
    result = np.concatenate((result, list_files(sms_dataset_path)))
    apps = list(map(process, result))
    
    for i, app in enumerate(apps):
        for type in app:
            type = list(type)
            for i in range(len(type)):
                for j in range(i, len(type)):
                    invoke_matrix[type[i]][type[j]] = 1

    pd.DataFrame(invoke_matrix, index=api_names, columns=api_names).to_csv(save_path)


def create_method(save_path: str):
    api_dataset = json.load(open(top_apis_path, "r"))[:num_top]
    api_names = [api['api_name'] for api in api_dataset]
    method_matrix = np.zeros((len(api_names), len(api_names)), dtype=np.int32)

    def process(app: str):
        in_app = []
        data = json.load(open(app, "r"))["data"]
        for method in data:
            buffer = []
            apis = method["api"]
            for api in apis:
                if api["full_api_call"] in api_names:
                    buffer.append(api_names.index(api["full_api_call"]))
            in_app.append(buffer)
        return in_app

    result = []
    result = np.concatenate((result, list_files(adware_dataset_path)))
    result = np.concatenate((result, list_files(banking_dataset_path)))
    result = np.concatenate((result, list_files(benign_dataset_path)))
    result = np.concatenate((result, list_files(riskware_dataset_path)))
    result = np.concatenate((result, list_files(sms_dataset_path)))
    
    apps = list(map(process, result))
    for i, app in enumerate(apps):
        for buffer in app:
            for i in range(len(buffer)):
                for j in range(i, len(buffer)):
                    method_matrix[buffer[i]][buffer[j]] = 1

    pd.DataFrame(method_matrix, index=api_names, columns=api_names).to_csv(save_path)


def create_package(save_path: str):
    api_dataset = json.load(open(top_apis_path, "r"))[:num_top]
    api_names = [api['api_name'] for api in api_dataset]
    package_matrix = np.zeros((len(api_names), len(api_names)), dtype=np.int32)
    
    for api_i in range(len(api_names)):
        package_matrix[api_i][api_i] = 1
        for api_j in range(api_i + 1, len(api_names)):
            if api_names[api_i][:api_names[api_i].index(';->')] == api_names[api_j][:api_names[api_j].index(';->')]:
                package_matrix[api_i][api_j] = 1
                package_matrix[api_j][api_i] = 1
    
    pd.DataFrame(package_matrix, index=api_names, columns=api_names).to_csv(save_path)


def create_custom_path_sim(app_api_path, invoke_path, package_path, method_path, save_path):
    app_api = pd.read_csv(app_api_path, index_col=0, header=0)
    label = app_api.pop('Label')
    app_api = app_api.to_numpy()
    invoke = pd.read_csv(invoke_path, index_col=0, header=0).to_numpy()
    package = pd.read_csv(package_path, index_col=0, header=0).to_numpy()
    method = pd.read_csv(method_path, index_col=0, header=0).to_numpy()

    agvSim = Path(app_api, invoke, package, method)

    save_int_csv(f"{save_path}/m1_{num_top}_APIs.csv", agvSim.meta_path_1())
    save_int_csv(f"{save_path}/m2_{num_top}_APIs.csv", agvSim.meta_path_2())
    save_int_csv(f"{save_path}/m3_{num_top}_APIs.csv", agvSim.meta_path_3())
    save_int_csv(f"{save_path}/m4_{num_top}_APIs.csv", agvSim.meta_path_4())


if __name__ == '__main__':
    base_path = f'./output/top_{num_top}_APIs'
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # create_app_api(f'{base_path}/app_api_{num_top}_APIs_train.csv')
    # create_invoke(f'{base_path}/invoke_{num_top}_APIs.csv')
    # create_method(f'{base_path}/method_{num_top}_APIs.csv')
    # create_package(f'{base_path}/package_{num_top}_APIs.csv')
    # create_custom_path_sim(f'{base_path}/app_api_{num_top}_APIs_train.csv',
    #                        f'{base_path}/invoke_{num_top}_APIs.csv',
    #                        f'{base_path}/package_{num_top}_APIs.csv',
    #                        f'{base_path}/method_{num_top}_APIs.csv',
    #                        f'{base_path}')

    create_app_api(f'{base_path}/app_api_{num_top}_APIs_test.csv')
