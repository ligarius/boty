import os
import subprocess


def test_run_celery_worker_falls_back_to_docker(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    log_file = tmp_path / "docker_exec.log"

    celery_script = bin_dir / "celery"
    celery_script.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"ModuleNotFoundError: No module named 'foo'\" >&2\n"
        "exit 1\n"
    )
    celery_script.chmod(0o755)

    docker_script = bin_dir / "docker"
    docker_script.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == compose ]]; then\n"
        "  shift\n"
        "  if [[ \"$1\" == version ]]; then\n"
        "    exit 0\n"
        "  fi\n"
        "  if [[ \"$1\" == exec ]]; then\n"
        "    shift\n"
        "    printf 'docker compose exec %s\\n' \"$*\" >> \"${RUNNER_LOG_FILE}\"\n"
        "    exit 0\n"
        "  fi\n"
        "fi\n"
        "echo 'unexpected docker usage' >&2\n"
        "exit 1\n"
    )
    docker_script.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["RUNNER_LOG_FILE"] = str(log_file)

    command = "source bot/scripts/_runner.sh; run_celery_worker -A app worker"
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=os.getcwd(),
        text=True,
        env=env,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    assert log_file.exists(), completed.stderr
    content = log_file.read_text()
    assert "exec worker celery -A app worker" in content
