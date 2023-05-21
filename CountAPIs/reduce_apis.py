import json


adware_apis = json.load(open("./output/All_APIs_in_Adware_600.json", "r"))
banking_apis = json.load(open("./output/All_APIs_in_Banking_600.json", "r"))
benign_apis = json.load(open("./output/All_APIs_in_Benign_600.json", "r"))
riskware_apis = json.load(open("./output/All_APIs_in_Riskware_600.json", "r"))
sms_apis = json.load(open("./output/All_APIs_in_SMS_600.json", "r"))

adware_api_names = set(adware_apis.keys())
banking_api_names = set(banking_apis.keys())
benign_api_names = set(benign_apis.keys())
riskware_api_names = set(riskware_apis.keys())
sms_api_names = set(sms_apis.keys())

all_api_names = adware_api_names & banking_api_names & riskware_api_names & sms_api_names & benign_api_names

all_apis = {}
min_app = 100

for api_name in all_api_names:
    if adware_apis[api_name]['app_num'] < min_app or banking_apis[api_name]['app_num'] < min_app or \
        riskware_apis[api_name]['app_num'] < min_app or sms_apis[api_name]['app_num'] < min_app or \
            benign_apis[api_name]['app_num'] < min_app:
        continue
    all_apis[api_name] = {
        'call_num': adware_apis[api_name]['call_num'] + banking_apis[api_name]['call_num'] + riskware_apis[api_name]['call_num'] + \
            sms_apis[api_name]['call_num'] + benign_apis[api_name]['call_num'],
        'app_num': adware_apis[api_name]['app_num'] + banking_apis[api_name]['app_num'] + riskware_apis[api_name]['app_num'] + \
            sms_apis[api_name]['app_num'] + benign_apis[api_name]['app_num']
    }

print(len(all_apis))
open(f"./output/APIs_in_All.json", "w").write(json.dumps(all_apis, indent=4))
