#!/usr/bin/env bash
# Starts the landslide web app (and optionally JupyterLab) with one command.
#   geostart        -> web app only
#   geostart lab    -> web app + JupyterLab
# Press Ctrl+C once to stop everything.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate geoai

# Stop a web app that's still running from last time
lsof -ti :8000 | xargs kill 2>/dev/null

cd ~/geoai/webapp
uvicorn app:app > ~/geoai/webapp.log 2>&1 &
APP_PID=$!
trap 'echo; echo "Stopping..."; kill $APP_PID 2>/dev/null; exit 0' INT TERM

# Wait until the app answers, then open it in the browser
for i in {1..30}; do
  curl -s -o /dev/null http://127.0.0.1:8000/api/runs && break
  sleep 1
done
open http://127.0.0.1:8000
echo "Web app running at http://127.0.0.1:8000  (log: ~/geoai/webapp.log)"

if [ "$1" = "lab" ]; then
  cd ~/geoai
  jupyter lab
  kill $APP_PID 2>/dev/null
else
  echo "Press Ctrl+C to stop."
  wait $APP_PID
fi
