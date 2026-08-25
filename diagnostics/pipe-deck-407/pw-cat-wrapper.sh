#!/bin/sh
# Diagnostic-only PATH-front wrapper for the real distro pw-cat.
set -u

: "${PIPE_DECK_PW_CAT_LOG:?PIPE_DECK_PW_CAT_LOG must name the wrapper log}"

{
    printf 'argv:'
    for argument in "$@"; do
        printf ' %s' "$argument"
    done
    printf '\n'
} >>"$PIPE_DECK_PW_CAT_LOG"

/usr/bin/pw-cat --verbose "$@" >>"$PIPE_DECK_PW_CAT_LOG" 2>&1
status=$?
printf 'exit=%s\n' "$status" >>"$PIPE_DECK_PW_CAT_LOG"
exit "$status"
