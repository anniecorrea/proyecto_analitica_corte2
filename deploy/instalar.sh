#!/usr/bin/env bash
# Instala (o actualiza) el frontend en un servidor Ubuntu (24.04 o 26.04) de AWS EC2, sin GPU. Ver deploy/AWS.md.
# Se puede correr las veces que haga falta: la primera vez instala todo y después solo trae los cambios.
#
#   curl -fsSL https://raw.githubusercontent.com/anniecorrea/proyecto_analitica_corte2/main/deploy/instalar.sh | bash
#   # o, con el repositorio ya clonado:
#   bash ~/proyecto_analitica_corte2/deploy/instalar.sh
#
# Variables opcionales: RAMA (por defecto main), REPO_URL, SWAP_GB (por defecto 4).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/anniecorrea/proyecto_analitica_corte2.git}"
RAMA="${RAMA:-main}"
SWAP_GB="${SWAP_GB:-4}"
PYTHON_VERSION="3.12"                     # la versión con la que se probó requirements-app.txt
DIR="$HOME/proyecto_analitica_corte2"

# Solo los archivos de Git LFS que usa app/inferencia.py. Bajar todos (~600 MB, SAM incluido) gasta la cuota
# mensual de LFS de GitHub y el disco del servidor sin necesidad.
LFS_INCLUIR="resultados/modelos/entrenamiento_multitarea/modelo_multitarea_final.pt"

echo "== Paquetes del sistema"
sudo apt-get update -y
sudo apt-get install -y git git-lfs curl
git lfs install --skip-repo

# Las instancias gratuitas tienen poca RAM; la swap evita que el proceso muera al leer una tomografía grande.
if [ ! -f /swapfile ]; then
    echo "== Swap de ${SWAP_GB} GB"
    sudo fallocate -l "${SWAP_GB}G" /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "== Código (rama $RAMA)"
if [ ! -d "$DIR/.git" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --branch "$RAMA" --depth 1 "$REPO_URL" "$DIR"
else
    git -C "$DIR" fetch --depth 1 origin "$RAMA"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$DIR" checkout -B "$RAMA" FETCH_HEAD
fi
git -C "$DIR" lfs pull --include="$LFS_INCLUIR"

echo "== Ambiente de Python ${PYTHON_VERSION}"
# El ambiente usa siempre Python 3.12, no el del sistema: Ubuntu 26.04 trae 3.14 y numpy 2.2.6 no tiene wheel para
# esa versión (pip intenta compilarlo y falla). uv descarga ese Python aparte, sin tocar el del sistema.
if ! command -v uv > /dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
if [ -x "$DIR/.venv/bin/python" ] && ! "$DIR/.venv/bin/python" -c \
        "import sys; sys.exit(sys.version_info[:2] != tuple(map(int, '${PYTHON_VERSION}'.split('.'))))"; then
    rm -rf "$DIR/.venv"                    # ambiente de un intento anterior con otra versión de Python
fi
[ -d "$DIR/.venv" ] || "$UV" venv --python "$PYTHON_VERSION" --seed "$DIR/.venv"
"$DIR/.venv/bin/pip" install --upgrade pip
"$DIR/.venv/bin/pip" install --no-cache-dir -r "$DIR/requirements-app.txt"

echo "== Túnel de Cloudflare"
if ! command -v cloudflared > /dev/null; then
    curl -fsSL -o /tmp/cloudflared.deb \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$(dpkg --print-architecture).deb"
    sudo dpkg -i /tmp/cloudflared.deb
    rm -f /tmp/cloudflared.deb
fi

echo "== Servicios"
for servicio in pengwin pengwin-tunel; do
    sed "s#/home/ubuntu#$HOME#g; s#^User=.*#User=$USER#" "$DIR/deploy/$servicio.service" \
        | sudo tee "/etc/systemd/system/$servicio.service" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable pengwin pengwin-tunel
sudo systemctl restart pengwin pengwin-tunel

IP=$(curl -fsS -m 3 https://checkip.amazonaws.com 2>/dev/null || echo "<IP-publica>")
URL=""
for _ in $(seq 30); do                  # el Quick Tunnel tarda unos segundos en dar su dirección
    URL=$(sudo journalctl -u pengwin-tunel --since "-2min" --no-pager | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' | tail -1 || true)
    [ -n "$URL" ] && break
    sleep 2
done
echo
echo "Listo (la primera carga del modelo tarda unos segundos)."
echo "  Túnel de Cloudflare (computador y celular): ${URL:-sudo journalctl -u pengwin-tunel | grep trycloudflare}"
echo "  Directo por IP:                             http://$IP:8501"
echo "Logs: sudo journalctl -u pengwin -f"