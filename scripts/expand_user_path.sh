#!/usr/bin/env bash
# Expand ~ / $VARS and strip inline comments / quotes from a .env value.
# Called from the Makefile so we never put bash ${var#prefix} (Make treats
# '#' as a comment) in a $(shell ...) line.
set -euo pipefail
python3 -c "import os, re, sys; q=chr(34)+chr(39); s=sys.argv[1].strip().strip(q); s=re.sub(r'\s+#.*$', '', s).strip(); print(os.path.expanduser(os.path.expandvars(s)))" "${1:-}"
