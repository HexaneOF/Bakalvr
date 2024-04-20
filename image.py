from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()

custom_objects = detector.CustomObjects(car=True, truck=True)
detections = detector.detectObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "car2.jpg"), output_image_path=os.path.join(execution_path , "image3custom.jpg"), minimum_percentage_probability=30, display_percentage_probability=False)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")