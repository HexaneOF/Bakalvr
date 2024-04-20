import os
import tkinter as tk
from tkinter import filedialog, Label

import numpy as np
import cv2
from PIL import Image, ImageTk
from imageai.Detection import ObjectDetection, VideoObjectDetection


EXECUTION_PATH = os.getcwd()


detector = ObjectDetection()
video_detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
video_detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(EXECUTION_PATH, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
video_detector.setModelPath(os.path.join(EXECUTION_PATH, "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
detector.loadModel()
video_detector.loadModel()
custom_objects = detector.CustomObjects(car=True, truck=True, bus=True)
video_custom_objects = video_detector.CustomObjects(car=True, truck=True, bus=True)



def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo
        global selected_image
        selected_image = np.array(image)


def process_image():
    if selected_image is not None:
        image, detections = detector.detectObjectsFromImage(
          custom_objects=custom_objects,
          input_image=selected_image,
          minimum_percentage_probability=1,
          display_percentage_probability=False,
          output_type='array',
        )
        image = Image.fromarray(image)
        global processed_image
        processed_image = image
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo


def download_image():
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")]
        )
        if file_path:
            processed_image.save(file_path)
            print(f"Изображение успешно сохранено как {file_path}")


def open_video_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        global video_path
        video_path = file_path


def process_video():
    if video_path is not None:
        output_path = filedialog.asksaveasfilename()
        video_detector.detectObjectsFromVideo(
            input_file_path=video_path,
            output_file_path=output_path,
            custom_objects=video_custom_objects,
            log_progress=True,
            display_percentage_probability=False
        )


# Создание главного окна
root = tk.Tk()
root.minsize(200, 200)
root.title("Робота")

# Создание кнопки для загрузки фото
button_load = tk.Button(root, text="Загрузити фото", command=open_file_dialog)
button_load.pack(pady=5)

# Создание кнопки для загрузки фото
button_load = tk.Button(root, text="Загрузити відео", command=open_video_dialog)
button_load.pack(pady=5)

# Создание кнопки для обработки фото
button_process = tk.Button(root, text="Обробка фото", command=process_image)
button_process.pack(pady=5)

button_process = tk.Button(root, text="Обробка відео", command=process_video)
button_process.pack(pady=5)

button_download = tk.Button(root, text="Скачать фото", command=download_image)
button_download.pack(pady=5)

# Создание метки для отображения выбранного изображения
label = tk.Label(root)
label.pack()

# Глобальная переменная для хранения выбранного изображения
selected_image = None
processed_image = None
video_path = None

root.mainloop()
