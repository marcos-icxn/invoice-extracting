from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image, ImageEnhance
import argparse
import base64
import json
import re
import tempfile
from dotenv import load_dotenv
import os

from services.notifier import enviar_factura

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY no definido")

DEFAULT_ENTRADA = "facturas"
DEFAULT_SALIDA = "resultados"

_CUIT_RE = re.compile(r'^\d{2}-\d{8}-\d$')


class Item(BaseModel):
    codigo: Optional[str] = None
    descripcion: str
    cantidad: float
    precio_unitario: float
    subtotal: float


class Proveedor(BaseModel):
    nombre: str
    cuit: str
    direccion: Optional[str] = None


class FacturaData(BaseModel):
    proveedor: Proveedor
    tipo_comprobante: str
    punto_venta: str
    numero: str
    fecha: str
    items: List[Item]
    subtotal: float
    iva: Optional[float] = None
    percepcion: Optional[float] = None
    retencion: Optional[float] = None
    total: float
    cae: Optional[str] = None
    observaciones: Optional[str] = None


def optimizar_imagen(ruta_entrada):
    print(f"  → Optimizando imagen...")

    img = Image.open(ruta_entrada)
    if img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')

    max_dimension = 1500
    ratio = max_dimension / max(img.size)
    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(img)
    img_enhanced = enhancer.enhance(1.3)

    fd, ruta_temp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img_enhanced.save(ruta_temp, 'JPEG', quality=85, optimize=True)

    return ruta_temp


def pdf_a_imagenes(ruta_pdf):
    if not PDF_SUPPORT:
        raise RuntimeError(
            "pdf2image no está instalado. Instalalo con: pip install pdf2image\n"
            "También necesitás poppler:\n"
            "  Linux:   sudo apt install poppler-utils  (o dnf/pacman según tu distro)\n"
            "  Windows: https://github.com/oschwartz10612/poppler-windows"
        )
    print(f"  → Convirtiendo PDF a imágenes...")
    paginas = convert_from_path(ruta_pdf, dpi=200)
    rutas = []
    for pagina in paginas:
        fd, ruta = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        pagina.save(ruta, 'JPEG', quality=90)
        rutas.append(ruta)
    return rutas


def extraer_datos_factura(ruta_imagen):
    print(f"  → Analizando con IA...")

    client = OpenAI(api_key=API_KEY)

    with open(ruta_imagen, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = """Extrae los datos de esta factura argentina en formato JSON.

IMPORTANTE SOBRE NÚMEROS:
- En Argentina, el PUNTO (.) se usa como separador de miles: 1.000 = mil
- La COMA (,) se usa como separador de decimales: 1.234,56 = mil doscientos treinta y cuatro con cincuenta y seis
- En el JSON, usá punto (.) para decimales según estándar JSON
- Ejemplos:
  * Si ves "1.234,56" → devolvé 1234.56
  * Si ves "10.000" → devolvé 10000
  * Si ves "999,50" → devolvé 999.50

Formato requerido:
{
  "tipo_comprobante": "Factura A/B/C",
  "punto_venta": "",
  "numero": "",
  "fecha": "DD/MM/AAAA",
  "proveedor": {
    "nombre": "",
    "cuit": "XX-XXXXXXXX-X",
    "direccion": ""
  },
  "items": [
    {
      "codigo": "",
      "descripcion": "",
      "cantidad": 0,
      "precio_unitario": 0,
      "subtotal": 0
    }
  ],
  "subtotal": 0,
  "iva": 0,
  "percepcion": 0,
  "retencion": 0,
  "total": 0,
  "cae": "",
  "observaciones": ""
}

IMPORTANTE: Responde únicamente con el JSON solicitado. Solo extrae datos que puedas leer claramente. Si algo no es legible, usa null. No expliques ni justifiques. No infieras valores."""

    response = client.responses.parse(
        model="gpt-5.4-nano",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            }
        ],
        text_format=FacturaData,
        max_output_tokens=4000,
    )

    datos = response.output_parsed
    print(f"  → ✓ Datos extraídos y validados")

    return datos


def validar_datos(datos):
    errores = []
    advertencias = []

    if not datos.numero:
        errores.append("❌ Falta número de factura")

    if not datos.proveedor.cuit:
        errores.append("❌ Falta CUIT del proveedor")
    elif not _CUIT_RE.match(datos.proveedor.cuit):
        advertencias.append(f"⚠️  CUIT con formato inválido: {datos.proveedor.cuit}")

    if datos.total is None:
        errores.append("❌ Falta total")

    if datos.items:
        subtotal_calculado = sum(item.subtotal for item in datos.items)
        subtotal_declarado = datos.subtotal

        if abs(subtotal_calculado - subtotal_declarado) > 0.01:
            advertencias.append(
                f"⚠️  Desigualdad: items=${subtotal_calculado:.2f} vs factura=${subtotal_declarado:.2f}"
            )

    return errores, advertencias


def procesar_factura(ruta_factura, carpeta_salida, nombre_override=None):
    nombre = nombre_override or os.path.basename(ruta_factura)
    print(f"\n📄 Procesando: {nombre}")
    print("=" * 70)

    ruta_opt = None
    try:
        ruta_opt = optimizar_imagen(ruta_factura)
        datos = extraer_datos_factura(ruta_opt)
        errores, advertencias = validar_datos(datos)

        resultado = {
            "archivo_original": nombre,
            "status": "error" if errores else ("warning" if advertencias else "ok"),
            "errores": errores,
            "advertencias": advertencias,
            "datos": datos.model_dump(),
        }

        print(f"\n  Status: {resultado['status'].upper()}")
        if errores:
            for error in errores:
                print(f"  {error}")
        if advertencias:
            for adv in advertencias:
                print(f"  {adv}")

        os.makedirs(carpeta_salida, exist_ok=True)
        nombre_salida = nombre.rsplit(".", 1)[0] + "_datos.json"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)

        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

        print(f"\n  💾 Guardado en: {ruta_salida}")

        if resultado["status"] != "error":
            print(f"  → Enviando datos al endpoint...")
            if enviar_factura(resultado):
                print(f"  ✓ Datos enviados correctamente")

        return resultado

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        return {
            "archivo_original": nombre,
            "status": "error",
            "errores": [str(e)],
            "datos": None,
        }

    finally:
        if ruta_opt and os.path.exists(ruta_opt):
            os.remove(ruta_opt)


def main():
    parser = argparse.ArgumentParser(description='Extractor de facturas argentinas con IA')
    parser.add_argument('--entrada', default=DEFAULT_ENTRADA, help='Carpeta con las facturas (default: facturas)')
    parser.add_argument('--salida', default=DEFAULT_SALIDA, help='Carpeta para los resultados (default: resultados)')
    args = parser.parse_args()

    carpeta_entrada = args.entrada
    carpeta_salida = args.salida

    print("\n" + "=" * 70)
    print("🤖 EXTRACTOR DE FACTURAS CON IA")
    print("=" * 70)

    if not os.path.exists(carpeta_entrada):
        print(f"\n❌ Error: La carpeta '{carpeta_entrada}' no existe")
        ext_str = "JPG, PNG, PDF" if PDF_SUPPORT else "JPG, PNG"
        print(f"Creá la carpeta y poné tus facturas ahí ({ext_str})")
        return

    extensiones = (".jpg", ".jpeg", ".png", ".pdf") if PDF_SUPPORT else (".jpg", ".jpeg", ".png")
    archivos = [
        f for f in os.listdir(carpeta_entrada)
        if f.lower().endswith(extensiones)
    ]

    if not archivos:
        ext_str = "JPG, PNG, PDF" if PDF_SUPPORT else "JPG, PNG"
        print(f"\n⚠️  No se encontraron facturas en '{carpeta_entrada}'")
        print(f"Formatos soportados: {ext_str}")
        return

    print(f"\n📁 Encontradas {len(archivos)} factura(s)\n")

    resultados = []
    for archivo in archivos:
        ruta = os.path.join(carpeta_entrada, archivo)

        if archivo.lower().endswith(".pdf"):
            rutas_img = []
            try:
                rutas_img = pdf_a_imagenes(ruta)
                nombre_base = archivo.rsplit(".", 1)[0]
                for i, ruta_img in enumerate(rutas_img):
                    nombre_pagina = f"{nombre_base}_p{i + 1}.jpg" if len(rutas_img) > 1 else f"{nombre_base}.jpg"
                    resultado = procesar_factura(ruta_img, carpeta_salida, nombre_override=nombre_pagina)
                    resultados.append(resultado)
            except Exception as e:
                print(f"\n  ❌ ERROR al convertir PDF '{archivo}': {e}")
                resultados.append({
                    "archivo_original": archivo,
                    "status": "error",
                    "errores": [str(e)],
                    "datos": None,
                })
            finally:
                for ruta_img in rutas_img:
                    if os.path.exists(ruta_img):
                        os.remove(ruta_img)
        else:
            resultado = procesar_factura(ruta, carpeta_salida)
            resultados.append(resultado)

    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    exitosos = sum(1 for r in resultados if r["status"] == "ok")
    con_warnings = sum(1 for r in resultados if r["status"] == "warning")
    con_errores = sum(1 for r in resultados if r["status"] == "error")

    print(f"  ✓ Exitosos: {exitosos}")
    print(f"  ⚠ Con advertencias: {con_warnings}")
    print(f"  ❌ Con errores: {con_errores}")
    print(f"\n  Resultados en: {carpeta_salida}/")
    print("\n¡Listo!")


if __name__ == "__main__":
    main()
