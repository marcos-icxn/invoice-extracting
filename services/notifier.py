import os
import time

import httpx

_WEBHOOK_URL = os.getenv("WEBHOOK_URL")
_WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

_TIMEOUT = 10.0
_MAX_RETRIES = 2
_RETRY_DELAYS = [1, 2]


def enviar_factura(payload: dict) -> bool:
    if not _WEBHOOK_URL:
        print("  ⚠️  WEBHOOK_URL no configurado, omitiendo envío")
        return False

    headers = {"Content-Type": "application/json"}
    if _WEBHOOK_SECRET:
        headers["Authorization"] = f"Bearer {_WEBHOOK_SECRET}"

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = httpx.post(
                _WEBHOOK_URL,
                json=payload,
                headers=headers,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPStatusError as e:
            print(f"  ⚠️  Webhook respondió {e.response.status_code}")
            if 400 <= e.response.status_code < 500:
                return False
        except httpx.RequestError as e:
            print(f"  ⚠️  Error de red al enviar webhook: {e}")

        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAYS[attempt])

    return False
