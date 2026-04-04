#!/usr/bin/env bash

# KERNEL: the name shown by `jupyter kernelspec list`
#   NOT a file path. To register your .venv:
#     source .venv/bin/activate
#     pip install ipykernel
#     python -m ipykernel install --user --name=myvenv
#   Then set KERNEL="myvenv"
KERNEL="${1:-python3}"

# REPO_DIR: root folder to start from. `find` recurses into ALL subfolders.
#   Use "." for current directory, or give an absolute/relative path.
REPO_DIR="${2:-.}"

# EXEC_TIMEOUT_SECONDS: nbconvert execution timeout in seconds.
#   Use a positive integer to apply both shell and nbconvert timeouts.
#   Use -1 to disable notebook execution timeout and rely on an outer job guard.
EXEC_TIMEOUT_SECONDS="${3:-3600}"

if [ "$EXEC_TIMEOUT_SECONDS" != "-1" ] && ! [[ "$EXEC_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: EXEC_TIMEOUT_SECONDS must be -1 or a positive integer." >&2
  exit 2
fi

find "$REPO_DIR" -name "*.ipynb" \
  ! -path "*/.ipynb_checkpoints/*" | sort | while read -r nb; do

  echo "▶ Running: $nb"
  if [ "$EXEC_TIMEOUT_SECONDS" = "-1" ]; then
    jupyter nbconvert \
      --to notebook \
      --execute "$nb" \
      --ExecutePreprocessor.kernel_name="$KERNEL" \
      --ExecutePreprocessor.timeout=-1 \
      --output /tmp/nb_test_out.ipynb \
      2>/tmp/nb_stderr.txt
  else
    timeout "$EXEC_TIMEOUT_SECONDS" jupyter nbconvert \
      --to notebook \
      --execute "$nb" \
      --ExecutePreprocessor.kernel_name="$KERNEL" \
      --ExecutePreprocessor.timeout="$EXEC_TIMEOUT_SECONDS" \
      --output /tmp/nb_test_out.ipynb \
      2>/tmp/nb_stderr.txt
  fi

  EXIT=$?
  if [ $EXIT -eq 0 ]; then
    echo "  ✓ PASS"
    echo "PASS: $nb" >> /tmp/nb_results.txt
  elif [ $EXIT -eq 124 ]; then
    echo "  ✗ TIMEOUT"
    echo "TIMEOUT: $nb" >> /tmp/nb_results.txt
  else
    echo "  ✗ FAIL"
    cat /tmp/nb_stderr.txt | tail -5
    echo "FAIL: $nb" >> /tmp/nb_results.txt
  fi
done

echo ""
echo "══════════ RESULTS ══════════"
grep "^PASS"    /tmp/nb_results.txt 2>/dev/null | sed 's/^/  ✓ /'
grep "^FAIL"    /tmp/nb_results.txt 2>/dev/null | sed 's/^/  ✗ /'
grep "^TIMEOUT" /tmp/nb_results.txt 2>/dev/null | sed 's/^/  ⧖ /'
echo "═════════════════════════════"
TOTAL=$(wc -l < /tmp/nb_results.txt)
PASSED=$(grep -c "^PASS" /tmp/nb_results.txt 2>/dev/null || echo 0)
echo "  $PASSED / $TOTAL passed"
rm -f /tmp/nb_results.txt /tmp/nb_stderr.txt /tmp/nb_test_out.ipynb
