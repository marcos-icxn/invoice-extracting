from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image, ImageEnhance
import base64
import json
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY no definido")

CARPETA_ENTRADA = "facturas"
CARPETA_SALIDA = "resultados"

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
    img_gray = img.convert('L')

    max_dimension = 1500
    ratio = max_dimension / max(img_gray.size)
    if ratio < 1:
        new_size = tuple(int(dim * ratio) for dim in img_gray.size)
        img_gray = img_gray.resize(new_size, Image.Resampling.LANCZOS)

    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(1.5)

    ruta_temp = "temp_optimizada.jpg"
    img_enhanced.save(ruta_temp, 'JPEG', quality=85, optimize=True)

    return ruta_temp


def extraer_datos_factura(ruta_imagen):
    """Extrae datos de la factura usando OpenAI"""
    print(f"  → Analizando con IA...")

    client = OpenAI(api_key=API_KEY)

    # Convertir imagen a base64
    with open(ruta_imagen, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Prompt para facturas argentinas
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

    # Llamada correcta a la API de OpenAI con structured outputs
    response = client.responses.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        response_format=FacturaData,
        max_tokens=4000,
    )

    datos = response.choices[0].message.parsed
    print(f"  → ✓ Datos extraídos y validados")

    return datos


def validar_datos(datos):
    """Valida los datos extraídos"""
    errores = []
    advertencias = []

    if not datos.numero:
        errores.append("❌ Falta número de factura")

    if not datos.proveedor.cuit:
        errores.append("❌ Falta CUIT del proveedor")

    if not datos.total:
        errores.append("❌ Falta total")

    # Validar cálculos
    if datos.items:
        subtotal_calculado = sum(item.subtotal for item in datos.items)
        subtotal_declarado = datos.subtotal

        if abs(subtotal_calculado - subtotal_declarado) > 0.01:
            advertencias.append(
                f"⚠️  Desigualdad: items=${subtotal_calculado:.2f} vs factura=${subtotal_declarado:.2f}"
            )

    return errores, advertencias


def procesar_factura(ruta_factura):
    """Procesa una factura completa"""
    nombre = os.path.basename(ruta_factura)
    print(f"\n📄 Procesando: {nombre}")
    print("=" * 70)

    try:
        # 1. Optimizar imagen
        ruta_opt = optimizar_imagen(ruta_factura)

        # 2. Extraer datos
        datos = extraer_datos_factura(ruta_opt)

        # 3. Validar
        errores, advertencias = validar_datos(datos)

        # 4. Preparar resultado (convertir Pydantic a dict)
        resultado = {
            "archivo_original": nombre,
            "status": "error" if errores else ("warning" if advertencias else "ok"),
            "errores": errores,
            "advertencias": advertencias,
            "datos": datos.model_dump(),  # Convertir Pydantic a dict
        }

        # 5. Mostrar resumen
        print(f"\n  Status: {resultado['status'].upper()}")
        if errores:
            for error in errores:
                print(f"  {error}")
        if advertencias:
            for adv in advertencias:
                print(f"  {adv}")

        # 6. Guardar resultado
        os.makedirs(CARPETA_SALIDA, exist_ok=True)
        nombre_salida = nombre.rsplit(".", 1)[0] + "_datos.json"
        ruta_salida = os.path.join(CARPETA_SALIDA, nombre_salida)

        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)

        print(f"\n  💾 Guardado en: {ruta_salida}")

        # Limpiar temporal
        if os.path.exists("temp_optimizada.jpg"):
            os.remove("temp_optimizada.jpg")

        return resultado

    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        return {
            "archivo_original": nombre,
            "status": "error",
            "errores": [str(e)],
            "datos": None,
        }


def main():
    """Función principal"""
    print("\n" + "=" * 70)
    print("🤖 EXTRACTOR DE FACTURAS CON IA")
    print("=" * 70)

    # Verificar carpeta
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"\n❌ Error: La carpeta '{CARPETA_ENTRADA}' no existe")
        print(f"Creá la carpeta y poné tus facturas ahí (JPG, PNG)")
        return

    # Buscar facturas
    archivos = [
        f
        for f in os.listdir(CARPETA_ENTRADA)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not archivos:
        print(f"\n⚠️  No se encontraron facturas en '{CARPETA_ENTRADA}'")
        return

    print(f"\n📁 Encontradas {len(archivos)} factura(s)\n")

    # Procesar cada una
    resultados = []
    for archivo in archivos:
        ruta = os.path.join(CARPETA_ENTRADA, archivo)
        resultado = procesar_factura(ruta)
        resultados.append(resultado)

    # Resumen final
    print("\n" + "=" * 70)
    print("📊 RESUMEN FINAL")
    print("=" * 70)
    exitosos = sum(1 for r in resultados if r["status"] == "ok")
    con_warnings = sum(1 for r in resultados if r["status"] == "warning")
    con_errores = sum(1 for r in resultados if r["status"] == "error")

    print(f"  ✓ Exitosos: {exitosos}")
    print(f"  ⚠ Con advertencias: {con_warnings}")
    print(f"  ❌ Con errores: {con_errores}")
    print(f"\n  Resultados en: {CARPETA_SALIDA}/")
    print("\n¡Listo!")


if __name__ == "__main__":
    main()