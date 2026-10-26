#!/usr/bin/env bash
# Copies the web app and your trained model files into a Hugging Face Space folder.
#
# Usage:  bash ~/geoai/deploy/prepare_space.sh ~/landslide-space
#   where ~/landslide-space is the Space you cloned with:  git clone https://huggingface.co/spaces/<you>/<space-name>
set -euo pipefail

SPACE="${1:?Give the path to your cloned Space folder, e.g. ~/landslide-space}"
GEOAI="${GEOAI_ROOT:-$HOME/geoai}"
HERE="$(cd "$(dirname "$0")" && pwd)"

[ -d "$SPACE/.git" ] || { echo "Error: $SPACE is not a git clone of your Space."; exit 1; }
[ -f "$GEOAI/landslide_unet.pt" ] || { echo "Error: $GEOAI/landslide_unet.pt not found. Train the model first (Step 4)."; exit 1; }

echo "Copying app files..."
cp "$GEOAI/webapp/app.py" "$GEOAI/webapp/pipeline.py" "$SPACE/"
mkdir -p "$SPACE/static"
cp "$GEOAI/webapp/static/index.html" "$SPACE/static/"
cp "$HERE/Dockerfile" "$HERE/requirements.txt" "$SPACE/"
cp "$HERE/SPACE_README.md" "$SPACE/README.md"

echo "Copying model files..."
mkdir -p "$SPACE/model"
for f in landslide_unet.pt model_config.json band_stats.npy band_quantiles.npy landslide_unet_v2.pt model_config_v2.json; do
  if [ -f "$GEOAI/$f" ]; then cp "$GEOAI/$f" "$SPACE/model/"; echo "  $f"; fi
done

cat > "$SPACE/.gitignore" <<'EOF'
model/runs/
__pycache__/
EOF

cd "$SPACE"
git lfs install --local >/dev/null
git lfs track "*.pt" "*.npy" >/dev/null
git add .gitattributes .gitignore README.md Dockerfile requirements.txt app.py pipeline.py static model

echo
echo "Ready. Now run:"
echo "  cd $SPACE"
echo "  git commit -m 'Deploy landslide detector'"
echo "  git push"
