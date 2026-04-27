#!/usr/bin/env bash
set -euo pipefail
#ESTA RUTINA AGARRA UNA PELI DE DINAMICAS ROTULADAS LOQUESEA.prmtop y QM_*.nc 
#Y TE ARMAR UN MILLON DE QM_*.rst7. O sea, te saca rst7s para cada frame de la peli.

#DECLARAR EL PRMTOP CORRECTO AQUI DEBAJO!
PRMTOP="${1:-1wla_sh.prmtop}"
NC_GLOB="${2:-QM_*.nc}"
START_INDEX="${3:-1}"

if ! command -v cpptraj >/dev/null 2>&1; then
  echo "Error: cpptraj no esta disponible en PATH." >&2
  exit 1
fi

if [[ ! -f "$PRMTOP" ]]; then
  echo "Error: no se encontro el archivo de topologia '$PRMTOP'." >&2
  exit 1
fi

mapfile -t NC_FILES < <(printf '%s\n' $NC_GLOB | sort -V)
if [[ ${#NC_FILES[@]} -eq 0 || "${NC_FILES[0]}" == "$NC_GLOB" ]]; then
  echo "Error: no se encontraron archivos con patron '$NC_GLOB'." >&2
  exit 1
fi

for f in "${NC_FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: '$f' no existe o no es archivo regular." >&2
    exit 1
  fi
done

if compgen -G 'QM_*.rst7' >/dev/null; then
  echo "Error: ya existen archivos QM_*.rst7 en este directorio." >&2
  echo "Borrarlos/moverlos antes de correr el script para evitar sobreescrituras." >&2
  exit 1
fi

TMPDIR="$(mktemp -d .qm_rst7_split_XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

counter="$START_INDEX"

for nc in "${NC_FILES[@]}"; do
  base="${nc%.nc}"
  in_file="$TMPDIR/${base}.in"

  cat > "$in_file" <<EOF
parm $PRMTOP
trajin $nc
trajout $TMPDIR/${base}.rst7 restart multi keepext
run
EOF

  cpptraj -i "$in_file" >/dev/null

  mapfile -t FRAMES < <(find "$TMPDIR" -maxdepth 1 -type f -name "${base}.*.rst7" | sort -V)
  if [[ ${#FRAMES[@]} -eq 0 ]]; then
    echo "Error: cpptraj no genero frames para '$nc'." >&2
    exit 1
  fi

  for frame_file in "${FRAMES[@]}"; do
    out_file="QM_${counter}.rst7"
    mv "$frame_file" "$out_file"
    counter=$((counter + 1))
  done
done

echo "Listo. Se generaron $((counter - START_INDEX)) archivos QM_*.rst7 (desde QM_${START_INDEX}.rst7)."
