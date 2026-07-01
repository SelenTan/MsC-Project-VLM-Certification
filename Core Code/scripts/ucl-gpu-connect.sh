#!/usr/bin/env bash
set -euo pipefail

host="${1:-ucl-gpu-london}"

case "$host" in
  ucl-gpu-medusa|ucl-gpu-cork|ucl-gpu-athens|ucl-gpu-london|ucl-gpu-geneva|ucl-gpu-turin|ucl-gpu-malmo|ucl-gpu-berlin)
    ;;
  *)
    echo "Unknown host alias: $host" >&2
    echo "Allowed: ucl-gpu-medusa, ucl-gpu-cork, ucl-gpu-athens, ucl-gpu-london, ucl-gpu-geneva, ucl-gpu-turin, ucl-gpu-malmo, ucl-gpu-berlin" >&2
    exit 2
    ;;
esac

read -r -p "UCL user ID (without @ucl.ac.uk): " userid

if [[ -z "$userid" ]]; then
  echo "UCL user ID is required." >&2
  exit 2
fi

exec ssh -l "$userid" "$host"
