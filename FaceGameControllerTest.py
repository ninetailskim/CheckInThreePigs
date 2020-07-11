from FaceGameController import FaceGameController

controller = FaceGameController(debug=True,use_gpu=False)

while True:
    controller.control()