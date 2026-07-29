#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bundle_root="$(cd "${script_dir}/.." && pwd)"
checksum_file="${bundle_root}/SHA256SUMS.txt"

if [[ ! -f "${checksum_file}" ]]; then
  echo "缺少 SHA256SUMS.txt" >&2
  exit 1
fi

(
  cd "${bundle_root}"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum --check SHA256SUMS.txt
  else
    shasum -a 256 -c SHA256SUMS.txt
  fi
)

if grep -R -I -n -E '/Users/|/home/[^ <]+|[A-Za-z]:\\\\Users\\\\|wxid_' \
  "${bundle_root}/simulation/README.md" \
  "${bundle_root}/simulation/RESULTS.md" \
  "${bundle_root}/simulation/configs" \
  "${bundle_root}/simulation/amrvac/spike_topping_solar_jet" \
  --exclude-dir=tests \
  --exclude='*.pyc'; then
  echo "发现可能的个人路径；请停止运行并检查。" >&2
  exit 1
fi

echo "校验和与公开文档隐私扫描通过。"
