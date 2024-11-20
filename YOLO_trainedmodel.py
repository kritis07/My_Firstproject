from ultralytics import YOLO #yolo model is imported from ultralytics framework after installation
model = YOLO("yolov8n.yaml")#we are using object detection model of YOLO category
results=model.train(data=  "C:/Users/hp/Downloads/phone_detectiondataset/data.yaml",imgsz=416,epochs=10)


