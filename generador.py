import qrcode
import json

datos = {
    "drawer_id": "DRW_001",
    "flight_number": "QR117",
    "total_drawer": 12,
    "drawer_category": "Breakfast",
    "customer_name": "Qatar Airways",
    "expiry_date": "2025-12-10",
}

data = json.dumps(datos, ensure_ascii=False, indent=2)

img = qrcode.make(data)
img.save("qr_producto3.png")
print("QR generado: qr_producto.png")

