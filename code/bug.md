# bug

height 總是記錯，但輸出到畫面時是對的

原因在 trainer.run()

```py
def run(net):
	init(net)
	test()
	for epoch in range(1, n_epochs + 1):
		train(epoch)
		test()
	model.save(net) # 這裡 train 完就 save 了，但是還沒設定 height ...
```

然後有些 save 的有 height ??? 但卻是錯的！

```json
{
  "filename": "9d039fbb009e855441a8a974ee5cc113e9b1db5e7ef5137bdda2e9ef180cb067",
  "model": {
    "in_shape": [
      28,
      28
    ],
    "out_shape": [
      10
    ],
    "layers": [
      {
        "type": "flatten"
      },
      {
        "type": "linear",
        "out_shape": [
          128
        ]
      },
      {
        "type": "relu"
      },
      {
        "type": "linear",
        "out_shape": [
          10
        ]
      }
    ],
    "parameter_count": 101770,
    "accuracy": 89.48999786376953,
    "height": 89.25500091552735
  }
}
```

因為 SolutionNet run 過才會有 height ???

```py
class SolutionNet(Solution):
    def height(self):
		net = self.net
		if not model.exist(net):
			trainer.run(net)
		else:
			jsonObj = model.load(net)
			net.model['accuracy'] = jsonObj['model']['accuracy']
		print('net=', net)
		h = net.model['accuracy']-(net.model['parameter_count']/10000)
		net.model['height'] = h
		print('model=', net.model)
		return h
```

結語： model.py 的模組定位有問題? 應重構！

model.py 是否應儲存 solutionNet 而非只是 Net
