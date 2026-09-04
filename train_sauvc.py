from ultralytics import YOLO


def main():
   
    model = YOLO("yolov8n.pt")

    model.train(
        data="sauvc.yaml",
        epochs=400,
        patience=100,      
        imgsz=960,         
        batch=16,
        device="mps",
        workers=0,
        cache=True,
        seed=0,            
        
        name="sauvc",

        
        hsv_h=0.02,        
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10.0,      
        scale=0.6,
        fliplr=0.5,
        flipud=0.0,        
        mosaic=1.0,
        close_mosaic=15,
    )

   
    metrics = model.val()
    print("mAP50-95:", round(metrics.box.map, 4))
    print("mAP50:   ", round(metrics.box.map50, 4))
    print("Веса и графики:", model.trainer.save_dir)


if __name__ == "__main__":
    main()
