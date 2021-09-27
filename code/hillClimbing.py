import sys


def log(s):
	file.write(s+"\n")

def hillClimbing(s, maxGens, maxFails):   # 爬山演算法的主體函數
	global file
	file = open('./model/hillClimbing.log', 'w')
	log(f"start: {s.str()}")              # 印出初始解
	fails = 0                             # 失敗次數設為 0
	# 當代數 gen<maxGen，且連續失敗次數 fails < maxFails 時，就持續嘗試尋找更好的解。
	for gens in range(maxGens):
		snew = s.neighbor()               #  取得鄰近的解
		sheight = s.height()              #  sheight=目前解的高度
		nheight = snew.height()           #  nheight=鄰近解的高度
		log(f'sheight:{sheight} nheight:{nheight}')
		if (nheight >= sheight):          #  如果鄰近解比目前解更好
			log(f'{gens}:{snew.str()}')  #    印出新的解
			s = snew                      #    就移動過去
			fails = 0                     #    移動成功，將連續失敗次數歸零
		else:                             #  否則
			fails = fails + 1             #    將連續失敗次數加一
		if (fails >= maxFails):
			log(f'fail {fails} times!')
			break
	log(f"solution: {s.str()}")          #  印出最後找到的那個解
	file.close()
	return s                              #    然後傳回。
