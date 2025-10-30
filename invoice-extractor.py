import base64
import json
import os

from openai import OpenAI
from PIL import Image, ImageEnhance

API_KEY = "apikey"
ENTRY_FOLDER = "facturas"
OUTPUT_FOLDER = "extracted_invoices"


def enhance_image(image_path):
    print(f" -> Optimizando imagen: {image_path}")

    img = Image.open(image_path)

    img_grayscale = img.convert("L")

    max_dimension = 1500
    ratio = max_dimension / max(img_grayscale.size)

    if ratio < 1:
        new_size = (
            int(img_grayscale.size[0] * ratio),
            int(img_grayscale.size[1] * ratio),
        )
        img_resized = img_grayscale.resize(new_size, Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(img_resized)
    img_enhanced = enhancer.enhance(1.5)

    temp_path = "temp_enhanced_image.jpg"
    img_enhanced.save(temp_path, format="JPEG", quality=85, optimize=True)

    print(f" -> Tamaño reducido: {img.size} -> {img_enhanced.size}")

    return temp_path


def extract_invoice_data(image_path):
    print(f" -> Enviando a OpenAI para análisis: {image_path}")

    client = OpenAI(api_key=API_KEY)

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = """Extrae los datos de esta factura argentina en formato JSON.
      IMPORTANTE: 
      - Solo extrae datos que puedas leer con certeza
      - Si algo no es legible, usa null
      - CUIT en formato: XX-XXXXXXXX-X
      - Fechas en formato: DD/MM/AAAA
      - Montos sin símbolo $, solo números

      Formato JSON:
      {
        "tipo_comprobante": "Factura A/B/C",
        "punto_venta": "",
        "numero": "",
        "fecha": "",
        "proveedor": {
          "nombre": "",
          "cuit": "",
          "direccion": ""
        },
        "items": [
          {
            "descripcion": "",
            "cantidad": 0,
            "precio_unitario": 0,
            "subtotal": 0
          }
        ],
        "subtotal": 0,
        "iva": 0,
        "total": 0,
        "cae": "",
        "observaciones": ""
      }"""
