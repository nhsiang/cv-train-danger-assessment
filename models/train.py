from ultralytics import YOLO

model = YOLO("./runs/semantic/train-2/weights/last.pt")

def main():
    model.train(data="railway.yaml", epochs=10, imgsz=1024, device=0)

if __name__ == "__main__":
    main()