from ultralytics import YOLO


def main():
    # предобученные веса; дообучаем под свои 9 классов
    model = YOLO("yolov8n.pt")

    model.train(
        data="sauvc.yaml",
        epochs=400,
        patience=100,      # стоп, если 100 эпох нет прогресса
        imgsz=960,         # крупнее вход -> лучше видно мелкие буи
        batch=16,
        device="mps",
        workers=0,
        cache=True,
        seed=0,            # было speed=0 - такого аргумента нет
        # project не задаём: путь берётся из глобального runs_dir
        # (yolo settings runs_dir=...), иначе получается runs/detect/runs/
        name="sauvc",

        # аугментации
        hsv_h=0.02,        # оттенок почти не трогаем: цвет = класс
        hsv_s=0.8,
        hsv_v=0.5,
        degrees=10.0,      # было deegrees
        scale=0.6,
        fliplr=0.5,
        flipud=0.0,        # перевёрнутых кадров под водой не будет
        mosaic=1.0,
        close_mosaic=15,
    )

    # val() внутри main(): переменная model локальная,
    # на уровне модуля её не видно -> был бы NameError
    metrics = model.val()
    print("mAP50-95:", round(metrics.box.map, 4))
    print("mAP50:   ", round(metrics.box.map50, 4))
    print("Веса и графики:", model.trainer.save_dir)


if __name__ == "__main__":
    main()
