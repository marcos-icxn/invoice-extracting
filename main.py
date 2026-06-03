import argparse
import base64
import datetime
import json
import os
import re
import tempfile
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageEnhance
from pydantic import BaseModel

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

_CUIT_RE = re.compile(r"^\d{11}$")


class Item(BaseModel):
    item: int
    code: Optional[str] = None
    description: str
    qty: float
    price: float
    percent: Optional[float] = None
    tax: Optional[float] = None


class FacturaData(BaseModel):
    cuit: str
    pos: int
    number: int
    letter: str
    date: str
    subtotal: float
    vat: Optional[float] = None
    vat_perception: Optional[float] = None
    vat_retention: Optional[float] = None
    iibb_perception: Optional[float] = None
    total: float
    cae: Optional[str] = None
    note: Optional[str] = None
    filename: Optional[str] = None
    items: List[Item]


def optimizar_imagen(ruta_entrada):
    print(f"  → Optimizando imagen...")

    img = Image.open(ruta_entrada)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    max_dimension = 1500
    ratio = max_dimension / max(img.size)
    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(img)
    img_enhanced = enhancer.enhance(1.3)

    fd, ruta_temp = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img_enhanced.save(ruta_temp, "JPEG", quality=85, optimize=True)

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
        pagina.save(ruta, "JPEG", quality=90)
        rutas.append(ruta)
    return rutas


def extraer_datos_factura(ruta_imagen):
    print(f"  → Analizando con IA...")

    client = OpenAI(api_key=API_KEY)

    with open(ruta_imagen, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    prompt = """Extrae los datos de esta factura argentina.

ITEMS — antes de extraer los ítems, identificá los encabezados de columna de la tabla (ej: Código, Descripción, Cantidad, Precio Unitario, Subtotal, IVA, etc.).
Luego leé cada fila estrictamente dentro de sus límites horizontales: cada valor pertenece al item de su propia fila, no a la fila anterior (la de arriba) ni a la siguiente (la de abajo).
  • qty: cantidad (columna "Cantidad" o "Cant.")
  • price: precio unitario (columna "Precio Unit." o similar) — NUNCA el precio es el subtotal de otra fila y TAMPOCO debe ser mayor al subtotal de su propia fila
  • en caso de que haya una columna de subtotal o importe por cada item (precio x cantidades de dicho item), la columna subtotal dividida las cantidades debe dar el precio
  • percent: alícuota de IVA del ítem (21, 10.5, 27 ó 0)
  • tax: monto de IVA del ítem, SOLO si hay una columna de IVA explícita en la tabla.
    NUNCA es el subtotal de línea (qty×price). Si no hay columna de IVA por línea, dejá tax en null.

TOTALES AL PIE:
  • subtotal: suma de subtotales netos de todos los ítems (qty×price). También llamado "Neto gravado" o "Base imponible".
  • vat: monto total de IVA.
  • vat_perception: percepciones de IVA — cargos adicionales cobrados por cuenta de organismos fiscales. Suman al total.
  • vat_retention: retenciones — montos retenidos por el comprador a cuenta de impuestos. Reducen el neto a pagar.
  • iibb_perception: percepción de IIBB -otro cargo adicional cobrado por cuenta de organismos fiscales provinciales. No suma al total pero va discriminado en caso que corresponda.
  • total: monto final de la factura.

Formato numérico argentino: punto (.) = miles, coma (,) = decimales.
Convertí al estándar JSON: "1.234,56"→1234.56 | "10.000"→10000 | "999,50"→999.50

Solo extrae lo claramente legible. Usá null para campos ausentes o ilegibles. No infieras valores. Si los montos finales tienen una desigualdad menor o igual a 10 centavos (0,10), no califica para advertencia."""

    response = client.responses.parse(
        model="gpt-5.4-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
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

    if not datos.number:
        errores.append("❌ Falta número de factura")

    if not datos.cuit:
        errores.append("❌ Falta CUIT del proveedor")
    elif not _CUIT_RE.match(datos.cuit):
        advertencias.append(f"⚠️  CUIT con formato inválido: {datos.cuit}")

    if datos.total is None:
        errores.append("❌ Falta total")

    if datos.items:
        subtotal_calculado = sum(item.qty * item.price for item in datos.items)
        if abs(subtotal_calculado - datos.subtotal) > 0.01:
            advertencias.append(
                f"⚠️  Desigualdad: items=${subtotal_calculado:.2f} vs factura=${datos.subtotal:.2f}"
            )

    return errores, advertencias


def _trim_resultados(carpeta, max_files=100):
    archivos = sorted(
        [
            os.path.join(carpeta, f)
            for f in os.listdir(carpeta)
            if f.endswith("_datos.json")
        ],
        key=os.path.getmtime,
    )
    for ruta in archivos[:-max_files]:
        os.remove(ruta)


def procesar_factura(ruta_factura, carpeta_salida, nombre_override=None):
    nombre = nombre_override or os.path.basename(ruta_factura)
    print(f"\n📄 Procesando: {nombre}")
    print("=" * 70)

    ruta_opt = None
    try:
        ruta_opt = optimizar_imagen(ruta_factura)
        datos = extraer_datos_factura(ruta_opt)
        datos.filename = nombre
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
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_salida = nombre.rsplit(".", 1)[0] + f"_{ts}_datos.json"
        ruta_salida = os.path.join(carpeta_salida, nombre_salida)

        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

        _trim_resultados(carpeta_salida)
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
    parser = argparse.ArgumentParser(
        description="Extractor de facturas argentinas con IA"
    )
    parser.add_argument(
        "--entrada",
        default=DEFAULT_ENTRADA,
        help="Carpeta con las facturas (default: facturas)",
    )
    parser.add_argument(
        "--salida",
        default=DEFAULT_SALIDA,
        help="Carpeta para los resultados (default: resultados)",
    )
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

    extensiones = (
        (".jpg", ".jpeg", ".png", ".pdf") if PDF_SUPPORT else (".jpg", ".jpeg", ".png")
    )
    archivos = [
        f for f in os.listdir(carpeta_entrada) if f.lower().endswith(extensiones)
    ]

    if not archivos:
        ext_str = "JPG, PNG, PDF" if PDF_SUPPORT else "JPG, PNG"
        print(f"\n⚠️  No se encontraron facturas en '{carpeta_entrada}'")
        print(f"Formatos soportados: {ext_str}")
        return

    ultimo = max(
        archivos, key=lambda f: os.path.getmtime(os.path.join(carpeta_entrada, f))
    )
    archivos = [ultimo]
    print(f"\n📁 Procesando última factura agregada: {ultimo}\n")

    resultados = []
    for archivo in archivos:
        ruta = os.path.join(carpeta_entrada, archivo)

        if archivo.lower().endswith(".pdf"):
            rutas_img = []
            try:
                rutas_img = pdf_a_imagenes(ruta)
                nombre_base = archivo.rsplit(".", 1)[0]
                for i, ruta_img in enumerate(rutas_img):
                    nombre_pagina = (
                        f"{nombre_base}_p{i + 1}.jpg"
                        if len(rutas_img) > 1
                        else f"{nombre_base}.jpg"
                    )
                    resultado = procesar_factura(
                        ruta_img, carpeta_salida, nombre_override=nombre_pagina
                    )
                    resultados.append(resultado)
            except Exception as e:
                print(f"\n  ❌ ERROR al convertir PDF '{archivo}': {e}")
                resultados.append(
                    {
                        "archivo_original": archivo,
                        "status": "error",
                        "errores": [str(e)],
                        "datos": None,
                    }
                )
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
