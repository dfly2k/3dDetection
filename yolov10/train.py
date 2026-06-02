from ultralytics import YOLOv10

# Load YOLOv10n model from scratch
#model = YOLOv10("weight\yolov10s.pt")
model = YOLOv10("yolov10s.yaml").load("../weights/yolov10s.pt")

# Train the model
model.train(data="../data/kitti.yaml", epochs=10, imgsz=640)