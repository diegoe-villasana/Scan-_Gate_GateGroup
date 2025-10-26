from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, json, os
from pyzbar.pyzbar import decode

app = FastAPI()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

drawer_capacity = {"DRW_001": 15, "DRW_002": 14, "DRW_007": 10}
drawer_current = {"DRW_001": 0, "DRW_002": 0, "DRW_007": 0}

ultimo_qr = ""
ultimo_qr_info = {
    "qr_data": None,
    "message": "Esperando QR...",
    "status": "waiting",
    "drawer": "",
    "current": 0,
    "capacity": 0
}

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def gen_frames():
    global ultimo_qr, ultimo_qr_info
    while True:
        success, frame = cap.read()
        if not success:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        codigos = decode(gray)

        for codigo in codigos:
            datos = codigo.data.decode("utf-8")
            if datos != ultimo_qr:
                try:
                    qr_info = json.loads(datos)
                    drawer = qr_info.get("drawer_id", "N/A")
                    ultimo_qr_info["qr_data"] = qr_info

                    if drawer not in drawer_current:
                        ultimo_qr_info.update({
                            "message": f"El drawer '{drawer}' no existe.",
                            "status": "error",
                            "drawer": drawer,
                            "current": 0,
                            "capacity": 0
                        })
                    elif drawer_current[drawer] < drawer_capacity[drawer]:
                        drawer_current[drawer] += 1
                        ultimo_qr_info.update({
                            "message": f"Producto agregado a {drawer}. Total: {drawer_current[drawer]}/{drawer_capacity[drawer]}",
                            "status": "ok",
                            "drawer": drawer,
                            "current": drawer_current[drawer],
                            "capacity": drawer_capacity[drawer]
                        })
                    else:
                        ultimo_qr_info.update({
                            "message": f"{drawer} ya está LLENO.",
                            "status": "full",
                            "drawer": drawer,
                            "current": drawer_current[drawer],
                            "capacity": drawer_capacity[drawer]
                        })
                    ultimo_qr = datos

                except json.JSONDecodeError:
                    ultimo_qr_info.update({
                        "qr_data": None,
                        "message": "El QR no trae un JSON válido.",
                        "status": "error",
                        "drawer": "",
                        "current": 0,
                        "capacity": 0
                    })
            x, y, w, h = codigo.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/scanner")
def scanner():
    return FileResponse(os.path.join(os.getcwd(), "scanner.html"))

@app.get("/ultimo_qr")
def obtener_qr(clear: bool = False):
    global ultimo_qr, ultimo_qr_info
    
    if clear:
        ultimo_qr = ""
        ultimo_qr_info.update({
            "qr_data": None,
            "message": "Esperando QR...",
            "status": "waiting",
            "drawer": "",
            "current": 0,
            "capacity": 0
        })
        
    return ultimo_qr_info
