#!/usr/bin/env bash

# Shared helpers for running Python modules or Celery workers either on the
# host (when dependencies are installed) or inside the Docker Compose
# containers that ship with the project.

get_compose_command() {
  if command -v docker-compose >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker-compose)
    return 0
  fi

  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE=(docker compose)
    return 0
  fi

  return 1
}

run_python_module() {
  local module="$1"
  shift

  local python_bin="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"

  if [[ -n "${python_bin:-}" ]]; then
    local tmp_err
    tmp_err=$(mktemp)
    set +e
    "$python_bin" -m "$module" "$@" 2> >(tee "$tmp_err" >&2)
    local status=$?
    set -e
    if [[ $status -eq 0 ]]; then
      rm -f "$tmp_err"
      return 0
    fi

    if [[ $status -eq 127 ]]; then
      rm -f "$tmp_err"
      if get_compose_command; then
        "${DOCKER_COMPOSE[@]}" exec api python -m "$module" "$@"
        return $?
      fi

      echo "Python interpreter '$python_bin' was not found and Docker Compose is unavailable. Install Python 3 with the project dependencies or run via Docker." >&2
      return $status
    fi

    if grep -Eq "ModuleNotFoundError|ImportError" "$tmp_err"; then
      rm -f "$tmp_err"
      if get_compose_command; then
        "${DOCKER_COMPOSE[@]}" exec api python -m "$module" "$@"
        return $?
      fi

      echo "Dependencies for '$module' are missing and Docker Compose was not found. Install requirements.txt or run inside the api container." >&2
      return $status
    fi

    rm -f "$tmp_err"
    return $status
  fi

  if get_compose_command; then
    "${DOCKER_COMPOSE[@]}" exec api python -m "$module" "$@"
    return $?
  fi

  echo "Python interpreter not found and Docker Compose is unavailable. Install Python 3 with the project dependencies or run via Docker." >&2
  return 127
}

run_celery_worker() {
  local worker_args=("$@")

  if command -v celery >/dev/null 2>&1; then
    local tmp_err
    tmp_err=$(mktemp)
    set +e
    celery "${worker_args[@]}" 2> >(tee "$tmp_err" >&2)
    local status=$?
    set -e

    if [[ $status -eq 0 ]]; then
      rm -f "$tmp_err"
      return 0
    fi

    if [[ $status -eq 127 ]] || grep -Eq "ModuleNotFoundError|ImportError" "$tmp_err"; then
      rm -f "$tmp_err"
      if get_compose_command; then
        "${DOCKER_COMPOSE[@]}" exec worker celery "${worker_args[@]}"
        return $?
      fi

      echo "Celery executable reported missing dependencies and Docker Compose is unavailable. Install requirements.txt or run inside the worker container." >&2
      return $status
    fi

    rm -f "$tmp_err"
    return $status
  fi

  if get_compose_command; then
    "${DOCKER_COMPOSE[@]}" exec worker celery "${worker_args[@]}"
    return $?
  fi

  echo "Celery executable not found and Docker Compose is unavailable. Install requirements.txt or run inside the worker container." >&2
  return 127
}
