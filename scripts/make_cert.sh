#!/usr/bin/env bash
# Self-signed HTTPS certificate for the RoadSense website, so phones on the same Wi-Fi can use the camera and GPS
# (browsers only allow location access on HTTPS pages or localhost).
# Covers localhost, 127.0.0.1 and this laptop's current LAN IPv4 addresses. Re-run when the laptop's IP changes.
#   bash scripts/make_cert.sh
set -euo pipefail
cd "$(dirname "$0")/.."
out=website/certs
mkdir -p "$out"
ips=$(ipconfig 2>/dev/null | grep -i "IPv4" | sed 's/.*: *//' | tr -d '\r' || true)
san="DNS:localhost,IP:127.0.0.1"
for ip in $ips; do san="$san,IP:$ip"; done
MSYS_NO_PATHCONV=1 openssl req -x509 -newkey rsa:2048 -nodes -sha256 -days 825 \
  -keyout "$out/key.pem" -out "$out/cert.pem" -subj "/CN=RoadSense local server" \
  -addext "subjectAltName=$san" -addext "extendedKeyUsage=serverAuth" 2>/dev/null
echo "Certificate for: $san"
echo "Saved $out/cert.pem and $out/key.pem (valid 825 days)"
for ip in $ips; do echo "Phone URL: https://$ip:8443"; done
