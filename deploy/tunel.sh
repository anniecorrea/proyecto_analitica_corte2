#!/usr/bin/env bash
# Lo arranca deploy/pengwin-tunel.service. Con TUNNEL_TOKEN (dominio propio en Cloudflare) corre ese túnel: cloudflared
# lee el token de la variable de entorno, así no aparece en la lista de procesos. Sin token, un Quick Tunnel.
if [ -n "${TUNNEL_TOKEN:-}" ]; then
    exec /usr/bin/cloudflared tunnel --no-autoupdate run
fi
exec /usr/bin/cloudflared tunnel --no-autoupdate --url http://localhost:8501
