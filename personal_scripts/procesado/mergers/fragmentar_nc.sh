#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Uso:
  ./fragmentar_nc.sh PRMTOP NC_O_PATRON [FOTOS] [BASE]

Ejemplos:
  ./fragmentar_nc.sh sistema.prmtop QM_01.nc
  ./fragmentar_nc.sh sistema.prmtop 'QM_*.nc' all
  ./fragmentar_nc.sh sistema.prmtop 'QM_*.nc' 277
  ./fragmentar_nc.sh sistema.prmtop 'QM_*.nc' 250-300
  ./fragmentar_nc.sh sistema.prmtop 'QM_*.nc' 250-end
  ./fragmentar_nc.sh sistema.prmtop 'QM_*.nc' 10,20,30-35

FOTOS acepta: all, N, A-B, N-end, o listas separadas por coma.
BASE acepta: vmd o amber. Por defecto BASE=vmd. VMD y cpptraj numeran 1--N.
Los rst7 salen como QM_<foto>.rst7 usando la misma numeracion de FOTOS.
Si CPPTRAJ esta definida, se usa ese binario. Si no, se prefiere $CONDA_PREFIX/bin/cpptraj.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 1
fi

PRMTOP="$1"
NC_GLOB="$2"
FRAME_SPEC="${3:-all}"
INDEX_BASE="${4:-vmd}"
CPPTRAJ="${CPPTRAJ:-}"

case "$INDEX_BASE" in
  vmd|amber) ;;
  *)
    echo "Error: BASE debe ser 'vmd' o 'amber'." >&2
    exit 1
    ;;
esac

if [[ -z "$CPPTRAJ" && -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/cpptraj" ]]; then
  CPPTRAJ="$CONDA_PREFIX/bin/cpptraj"
elif [[ -z "$CPPTRAJ" ]]; then
  CPPTRAJ="$(command -v cpptraj || true)"
fi

if [[ -z "$CPPTRAJ" || ! -x "$CPPTRAJ" ]]; then
  echo "Error: cpptraj no esta disponible en PATH." >&2
  exit 1
fi

if [[ ! -f "$PRMTOP" ]]; then
  echo "Error: no se encontro el archivo de topologia '$PRMTOP'." >&2
  exit 1
fi

mapfile -t NC_FILES < <(compgen -G "$NC_GLOB" | sort -V)
if [[ ${#NC_FILES[@]} -eq 0 ]]; then
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

FRAME_MODE="selected"
CPP_ONLYFRAMES=""
OUT_LABELS=()
TOTAL_FRAMES=""
declare -A SEEN_LABELS=()

get_total_frames() {
  local output frames
  local cmd=("$CPPTRAJ" -p "$PRMTOP")

  for nc in "${NC_FILES[@]}"; do
    cmd+=(-y "$nc")
  done
  cmd+=(-tl)

  output="$("${cmd[@]}")"
  frames="$(awk '/^Frames:/ {print $2; exit}' <<< "$output")"
  if [[ ! "$frames" =~ ^[0-9]+$ || "$frames" -lt 1 ]]; then
    echo "Error: no pude determinar la cantidad de frames con cpptraj -tl." >&2
    echo "$output" >&2
    exit 1
  fi

  TOTAL_FRAMES="$frames"
}

add_frame() {
  local label="$1"
  local cpp_frame
  local out_label

  if [[ -n "${SEEN_LABELS[$label]:-}" ]]; then
    echo "Error: la foto '$label' esta repetida en la seleccion." >&2
    exit 1
  fi
  SEEN_LABELS[$label]=1

  if [[ "$label" -lt 1 ]]; then
    echo "Error: las fotos empiezan en 1." >&2
    exit 1
  fi
  cpp_frame="$label"
  out_label="$label"

  if [[ -n "$TOTAL_FRAMES" && "$cpp_frame" -gt "$TOTAL_FRAMES" ]]; then
    echo "Error: la foto '$label' no existe. La trayectoria tiene $TOTAL_FRAMES frames." >&2
    exit 1
  fi

  OUT_LABELS+=("$out_label")
  CPP_FRAMES+=("$cpp_frame")
}

parse_frame_spec() {
  local spec="$1"
  local part start end frame
  CPP_FRAMES=()

  if [[ "$spec" == "all" || "$spec" == "todas" || "$spec" == "*" ]]; then
    FRAME_MODE="all"
    return
  fi

  get_total_frames

  IFS=',' read -r -a PARTS <<< "$spec"
  for part in "${PARTS[@]}"; do
    if [[ "$part" =~ ^[0-9]+$ ]]; then
      add_frame "$part"
    elif [[ "$part" =~ ^([0-9]+)-end$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="$TOTAL_FRAMES"
      if (( start > end )); then
        echo "Error: rango invalido '$part'. La trayectoria tiene $TOTAL_FRAMES frames." >&2
        exit 1
      fi
      for ((frame = start; frame <= end; frame++)); do
        add_frame "$frame"
      done
    elif [[ "$part" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      end="${BASH_REMATCH[2]}"
      if (( start > end )); then
        echo "Error: rango invalido '$part'." >&2
        exit 1
      fi
      for ((frame = start; frame <= end; frame++)); do
        add_frame "$frame"
      done
    else
      echo "Error: seleccion de fotos invalida '$part'." >&2
      usage >&2
      exit 1
    fi
  done

  if [[ ${#CPP_FRAMES[@]} -eq 0 ]]; then
    echo "Error: no se selecciono ninguna foto." >&2
    exit 1
  fi

  local old_ifs="$IFS"
  IFS=,
  CPP_ONLYFRAMES="${CPP_FRAMES[*]}"
  IFS="$old_ifs"
}

parse_frame_spec "$FRAME_SPEC"

TMPDIR="$(mktemp -d .qm_rst7_split_XXXXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

in_file="$TMPDIR/cpptraj.in"
out_prefix="$TMPDIR/frames.rst7"

{
  printf 'parm %s\n' "$PRMTOP"
  for nc in "${NC_FILES[@]}"; do
    printf 'trajin %s\n' "$nc"
  done
  if [[ "$FRAME_MODE" == "all" ]]; then
    printf 'trajout %s restart multi keepext\n' "$out_prefix"
  else
    printf 'trajout %s restart multi keepext onlyframes %s\n' "$out_prefix" "$CPP_ONLYFRAMES"
  fi
  printf 'run\n'
} > "$in_file"

"$CPPTRAJ" -i "$in_file" > "$TMPDIR/cpptraj.out" 2>&1

if [[ -f "$out_prefix" ]]; then
  FRAMES=("$out_prefix")
else
  mapfile -t FRAMES < <(find "$TMPDIR" -maxdepth 1 -type f -name 'frames.*.rst7' | sort -V)
fi

if [[ ${#FRAMES[@]} -eq 0 ]]; then
  echo "Error: cpptraj no genero frames." >&2
  echo "Ultimas lineas de cpptraj:" >&2
  tail -n 20 "$TMPDIR/cpptraj.out" >&2
  exit 1
fi

if [[ "$FRAME_MODE" == "selected" && ${#FRAMES[@]} -ne ${#OUT_LABELS[@]} ]]; then
  echo "Error: cpptraj genero ${#FRAMES[@]} frames, pero se pidieron ${#OUT_LABELS[@]}." >&2
  echo "Revisar si la seleccion '$FRAME_SPEC' existe en la trayectoria." >&2
  exit 1
fi

if [[ "$FRAME_MODE" == "all" ]]; then
  counter=1

  for frame_file in "${FRAMES[@]}"; do
    mv "$frame_file" "QM_${counter}.rst7"
    counter=$((counter + 1))
  done
else
  for i in "${!FRAMES[@]}"; do
    mv "${FRAMES[$i]}" "QM_${OUT_LABELS[$i]}.rst7"
  done
fi

echo "Listo. Se generaron ${#FRAMES[@]} archivos QM_*.rst7 usando numeracion BASE=$INDEX_BASE."
