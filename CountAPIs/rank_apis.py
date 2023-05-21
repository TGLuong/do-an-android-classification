import json


apis_in_all = json.load(open(f"./output/APIs_in_All.json", "r"))
apis = []
for key in apis_in_all.keys():
    apis.append({
        'api_name': key,
        'call_num': apis_in_all[key]['call_num'],
        'app_num': apis_in_all[key]['app_num'],
    })
apis.sort(key=lambda x: x["app_num"], reverse=True)
open(f"./output/Sorted_APIs_in_All.json", "w").write(json.dumps(apis, indent=4))
