#!/usr/bin/env bash
# --------------------------------------------------------
# run-api.sh – start the Flask application with configuration
#               provided via environment variables.
# --------------------------------------------------------

# ---- 1. Default values ------------------------------------
# If the variables are not already defined in the environment,
# fall back to the same defaults used inside the Python code.

# Port on which the Flask server will listen
export ANON_API_PORT="${ANON_API_PORT:-5001}"

# Flask debug flag (true/false)
# Accepts "true", "1", "yes" (case‑insensitive) as true,
# everything else is treated as false.
if [[ -z "$ANON_API_DEBUG" ]]; then
    export ANON_API_DEBUG="false"
else
    case "${ANON_API_DEBUG,,}" in
        1|true|yes) export ANON_API_DEBUG="true" ;;
        *)           export ANON_API_DEBUG="false" ;;
    esac
fi

# ---- 2. Informational output ------------------------------
echo "--------------------------------------------------------"
echo "Starting PII Classification API"
echo "Port      : $ANON_API_PORT"
echo "Debug mode: $ANON_API_DEBUG"
echo "--------------------------------------------------------"

# ---- 3. Launch the application -----------------------------
# The module `pii_classification.api.app` defines the Flask app
# and is executed with `python -m …`. The app will read the
# environment variables set above at startup.

python3 -m pii_classification.api.app