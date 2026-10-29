import modal

app = modal.App("arc3")

gpu = "RTX-PRO-6000"
d = "/repo/ARC3-Inference"

# make sbatch GAME=[] GAME_TAGS=official MAX_EXPERIMENT_RUNTIME_MINUTES=360
# make interactive GAME=taps N_PASSES=1 MAX_RUNTIME_MINUTES=10
run_name = "baseline-qwen"
game = "tn36-ef4dde99"
n_passes = 1
max_runtime_mins = 10

arc3_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git", "make", "curl", "build-essential")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/Decrescent398/arc-agi-3-solution.git /repo", 
        f"chmod +x {d}/scripts/*.sh",
        f"cd {d} && uv lock && make install")
)

hf_secrets = modal.Secret.from_name("huggingface")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results = modal.Volume.from_name("arc3-results", create_if_missing=True)

common = dict(image=arc3_image, 
              volumes={
                  "/results": results, 
                  "/root/.cache/huggingface": hf_cache,
                  },
              secrets=[hf_secrets],
            )


@app.function(**common, gpu=gpu, timeout=3600*10)
def run():
    import subprocess
    
    # server = subprocess.Popen(f"cd {d} && make server", shell=True)
    # subprocess.run(f"cd {d} && make check-server", check=True, shell=True)
    
    try:
        subprocess.run(
            f"cd {d} && uv run --no-sync python -c \"import arc_agi; arc_agi.Arcade(environments_dir='/results/envs').make('{game}')\"",
            check=True, shell=True)

        subprocess.run(
            f"cd {d} && make server && make interactive GAME={game} N_PASSES={n_passes} "
            f"MAX_RUNTIME_MINUTES={max_runtime_mins} ENVIRONMENTS_DIR=/results/envs EXPERIMENTS_DIR=/results",
            check=True, shell=True)
        
    # subprocess.run(f"cd {d} && make stop-server")
    # server.terminate()
    finally:
        results.commit()
        hf_cache.commit()
        
# @app.function(**common)
# def list_games():
#     import subprocess
#     subprocess.run(f"cd {d} && uv run --no-sync inference-taaf-run --include-tags official --list-games", shell=True, check=True)