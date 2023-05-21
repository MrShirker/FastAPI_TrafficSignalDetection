from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')

import torch
import random

import ultralytics
import os

from sort import Sort

app = FastAPI()

app.mount("/gif", StaticFiles(directory="gif"), name="gif")

model_selection_options = ['yolov5s','yolov5m','yolov5l','yolov5x','yolov5n',
                        'yolov5n6','yolov5s6','yolov5m6','yolov5l6','yolov5x6']
model_dict = {model_name: None for model_name in model_selection_options} #set up model cache

# Obtener la lista de nombres de archivos en la carpeta
file_names = os.listdir('modelos/')
# Extensiones de modelos válidas
valid_extensions = ['.torchscript', '.onnx', '_openvino_model', '.engine ', '.mlmodel', '_saved_model', '.pt', '.tflite', '_edgetpu.tflite', '_paddle_model ']

# Filtrar solo los archivos con extensiones de modelos válidas
model_names = [file_name for file_name in file_names if os.path.splitext(file_name)[1] in valid_extensions]
# Crear el diccionario con los nombres de los modelos
custom_model_dict = {model_name: None for model_name in model_names}

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

# Crear la instancia de SORT
mt_tracker = Sort()

##############################################
#-------------GET Request Routes--------------
##############################################
@app.get("/", response_class=HTMLResponse)
async def root():
    gif_files = os.listdir("gif")  # Obtiene la lista de archivos en la carpeta "gif"
    random_gif = random.choice(gif_files)  # Elige un GIF aleatorio de la lista
    message = "<h1>Bienvenido la página del Proyecto de Coche autónomo del Grupo TP2.2</h1>"
    gif_url = f"/gif/{random_gif}"  # Construye la URL del GIF seleccionado

    html_content = f"""
    <html>
        <head>
            <title>Página principal</title>
        </head>
        <body>
            {message}
            <img src="{gif_url}" alt="GIF aleatorio">
        </body>
    </html>
    """

    return html_content


##############################################
#------------POST Request Routes--------------
##############################################
@app.post("/detect")
def detect_via_api(request: Request,
                file_list: List[UploadFile] = File(...), 
                model_name: str = Form(...),
                img_size: Optional[int] = Form(640),
                tracking: Optional[bool] = Form(False)):
    
    '''
    Requires an image file upload, model name (ex. yolov5s). 
    Optional image size parameter (Default 640)
    Optional download_image parameter that includes base64 encoded image(s) with bbox's drawn in the json response
    
    Returns: JSON results of running YOLOv5 on the uploaded image. Bbox format is X1Y1X2Y2. 
            If download_image parameter is True, images with
            bboxes drawn are base64 encoded and returned inside the json response.

    Intended for API usage.
    '''
    
    img_batch = [cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB, 
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]
    
    if model_name in custom_model_dict:
        custom_model = True
    else:
        custom_model = False
        
    if model_name in model_dict:
        yolo_model = True
    else:
        yolo_model = False

    if custom_model:
        if custom_model_dict[model_name] is None:
            custom_path = 'modelos/'+model_name
            custom_model_dict[model_name] = torch.hub.load('ultralytics/yolov5', 'custom',  path=custom_path)
        
        # Obtener las detecciones
        results = custom_model_dict[model_name](img_batch_rgb, size = img_size) 
        json_results = results_to_json(results,custom_model_dict[model_name])
        
        if tracking:
            detections = results.pred[0].numpy()
            # Actualizar SORT
            track_bbs_ids = mt_tracker.update(detections)
            
            if len(track_bbs_ids) > 0:
                for j in range(len(track_bbs_ids.tolist())):
                    coords = track_bbs_ids.tolist()[j]
                    
                    # Agregar el ID actualizado a json_results
                    json_results[0][j]['tracker_id'] = int(coords[4])  
    elif yolo_model:
        if model_dict[model_name] is None:
            model_dict[model_name] = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        
        # Obtener las detecciones
        results = model_dict[model_name](img_batch_rgb, size = img_size) 
        json_results = results_to_json(results,model_dict[model_name])
        
        if tracking:
            detections = results.pred[0].numpy()
            # Actualizar SORT
            track_bbs_ids = mt_tracker.update(detections)
            
            if len(track_bbs_ids) > 0:
                for j in range(len(track_bbs_ids.tolist())):
                    coords = track_bbs_ids.tolist()[j]
                    
                    # Agregar el ID actualizado a json_results
                    json_results[0][j]['tracker_id'] = int(coords[4])     
    else:
        print("El modelo elegido no esta disponible")

    encoded_json_results = str(json_results).replace("'",r'"')
    return encoded_json_results

@app.get("/custom_models")
def get_custom_models():    
    lista = list(custom_model_dict)
    encoded_json_results = str(lista).replace("'",r'"')
    return encoded_json_results
    
##############################################
#--------------Helper Functions---------------
##############################################

def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
            ]


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default = 'localhost')
    parser.add_argument('--port', default = 8000)
    parser.add_argument('--precache-models', action='store_true', 
            help='Pre-cache all models in memory upon initialization, otherwise dynamically caches models')
    opt = parser.parse_args()

    if opt.precache_models:
        model_dict = {model_name: torch.hub.load('ultralytics/yolov5', model_name, pretrained=True) 
                        for model_name in model_selection_options}
    
    app_str = 'main:app' #make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host= opt.host, port=opt.port, reload=True)
