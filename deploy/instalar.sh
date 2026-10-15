#!/usr/bin/env bash
# Instala (o actualiza) el frontend en un servidor Ubuntu 24.04 de AWS EC2, sin GPU. Ver deploy/AWS.md.
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
DIR="$HOME/proyecto_analitica_corte2"

# Solo los archivos de Git LFS que usa app/inferencia.py. Bajar todos (~600 MB, SAM incluido) gasta la cuota
# mensual de LFS de GitHub y el disco del servidor sin necesidad.
LFS_INCLUIR="resultados/modelos/entrenamiento_multitarea/modelo_multitarea_final.pt"

echo "== Paquetes del sistema"
sudo apt-get update -y
sudo apt-get install -y git git-lfs python3-venv python3-pip
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

echo "== Ambiente de Python"
[ -d "$DIR/.venv" ] || python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install --upgrade pip
"$DIR/.venv/bin/pip" install --no-cache-dir -r "$DIR/requirements-app.txt"

echo "== Servicio"
sed "s#/home/ubuntu#$HOME#g; s#^User=.*#User=$USER#" "$DIR/deploy/pengwin.service" \
    | sudo tee /etc/systemd/system/pengwin.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable pengwin
sudo systemctl restart pengwin

IP=$(curl -fsS -m 3 https://checkip.amazonaws.com 2>/dev/null || echo "<IP-publica>")
echo
echo "Listo. Abre http://$IP:8501 (la primera carga del modelo tarda unos segundos)."
echo "Logs: sudo journalctl -u pengwin -f"