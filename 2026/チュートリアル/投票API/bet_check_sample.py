import requests
import json

LOGIN_API_URL  = "https://masters.netkeiba.com/ai2026_student/api/login"
BET_API_URL    = "https://masters.netkeiba.com/ai2026_student/api/bet"

LOGIN_ID       = ""
LOGIN_PASSWORD = ""
RACE_ID        = ""

# 投票内容確認API用パラメータ準備
def buildQueryForRaceIds(bet_data) :
	params = {}
	index = 0
	for bet_items in bet_data:
		for key in bet_items:
			if key != 'race_id' :
				continue
			param_key = "race_id[{}]".format(index)
			params[param_key] = bet_items[key]
			index += 1
	return params

# 共通処理:APIレスポンス取得
def getResponseData(json_response) :
	rj = json.dumps(json_response.json())
	rj_dec = json.loads(rj)

	# 成功時は rj_dec にキー status が存在し rj_dec["status"] が"OK"
	ret = 'status' in rj_dec.keys()
	if ret == False :
		return False
	status = rj_dec["status"]
	if status != 'OK':
		return False
	# 成功時は rj_dec にキー data が存在
	ret = 'data' in rj_dec.keys()
	if ret == False :
		return False
	return rj_dec["data"]

# ログイン
def execLogin(login_id,password) :
	url = LOGIN_API_URL
	data = {"login_id" : login_id , "password" : password}
	r = requests.post(url, data)
	response_data = getResponseData(r)
	if response_data == False :
		return False
	accessToken = response_data["access_token"]
	return accessToken

# 投票内容確認
def checkBet(accessToken,dataBet) :
	url = BET_API_URL
	params = buildQueryForRaceIds(dataBet["bet_data"])
	auth_headers = {
	    'Content-Type' : 'application/json',
	    'Authorization' : 'Bearer '+ accessToken
	}
	r = requests.get(url, params=params ,headers=auth_headers) 
	response_data = getResponseData(r)
	if response_data == False :
		return False
	return response_data

# ----------------------
# (1)ログイン
# (アクセストークンを発行します。トークンの有効期限は5分間です)
# ----------------------
login_id = LOGIN_ID
password = LOGIN_PASSWORD
accessToken = execLogin(login_id,password)
if accessToken == False :
	print("[Login]:Failed")
	exit()
else:
	print("[Login]:Success")
	print(accessToken)

# ----------------------
# (2)投票内容の確認
# (投票処理に成功していれば投票結果を取得します)
# ----------------------
dataBet = { "bet_data" : [ { "race_id" : RACE_ID } ] }
resultCheckBet = checkBet(accessToken,dataBet)
if resultCheckBet == False :
	print("[Bet](Check):Failed")
	exit()
else:
	print("[Bet](Check):Success")
	print(resultCheckBet)
