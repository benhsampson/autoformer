# -e : Exit immediately if a command exits with a non-zero status.
# -E : If set, the ERR trap is inherited by shell functions.
# -u : Treat unset variables as an error when substituting.
set -eEu
set -o pipefail # Fail a pipe if any sub-command fails.
set -x # Print commands and their arguments as they are executed.

python run.py
