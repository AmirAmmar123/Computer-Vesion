from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch


results = model.train(data='config.yaml', epochs=1)  # predict on an image
