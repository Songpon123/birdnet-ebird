#!/bin/zsh
# zip ของโปรแกรมสำหรับส่งให้เพื่อน -> Desktop/BirdNET-eBird-mac-share.zip
# ใส่โมเดล BirdNET 3.0 (data/birdnet) ไปด้วย เพื่อนไม่ต้องโหลดเอง
# ไม่ใส่: .git (ประวัติ GitHub), data/settings.json (API key), analysis-presets.json (สถานที่), field-logs.json (checklist ของคุณ), reference-sounds (ไฟล์ของคุณ), cache, backup, tests
set -eu
program="$(cd "$(dirname "$0")/.." && pwd)"
name="$(basename "$program")"
out="${1:-$HOME/Desktop/$name-share.zip}"      # หรือระบุที่เก็บเอง
rm -f "$out"
cd "$program/.."
/usr/bin/zip -r -y -q "$out" "$name" \
    -x "$name/data/settings.json*" "$name/data/analysis-presets.json*" "$name/data/field-logs.json*" "$name/data/cache/*" "$name/data/reference-sounds/*" \
       "$name/data/.write-test" "$name/_backup-before-upgrade/*" "$name/tests/*" \
       "$name/accuracy_cases/*" "$name/build/*" "$name/.git/*" "$name/.gitignore" "*/__pycache__/*" "*.pyc" "*/.DS_Store"
if /usr/bin/unzip -l "$out" | grep -q "data/settings.json"; then
    rm -f "$out"; echo "ERROR: settings.json slipped in; zip removed" >&2; exit 1
fi
echo "Created $out ($(du -h "$out" | cut -f1)) — API keys not included"
