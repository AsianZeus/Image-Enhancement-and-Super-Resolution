import cv2
import os

# modelName = "fsrcnn"
modelName = "edsr"
modelScale = 3

sr = cv2.dnn_superres.DnnSuperResImpl_create()
# sr.readModel("FSRCNN_x3.pb")
sr.readModel("EDSR_x4.pb")
sr.setModel(modelName, modelScale)

for i in os.listdir("images"):
    name,extension=i.split('.')
    print(name,extension)
    image = cv2.imread(f"images/{i}")
    upscaled = sr.upsample(image)
    cv2.imwrite(name+"_upscaled."+extension,upscaled)