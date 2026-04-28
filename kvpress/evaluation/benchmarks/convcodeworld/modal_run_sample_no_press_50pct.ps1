# Sample run: shard 0 of the 50% tune split (57 tasks), full KV cache (no_press).
# Run from kvpress/ directory in PowerShell:
#
#   cd C:\Users\Prodyut\Downloads\TurnKV\kvpress
#   .\evaluation\benchmarks\convcodeworld\modal_run_sample_no_press_50pct.ps1
#
# Prerequisites:
#   1. Modal CLI installed and authenticated (modal token new / modal profile current)
#   2. HF token set up as a Modal secret:
#      modal secret create hf-secret HF_TOKEN=hf_YOUR_TOKEN_HERE
#      (or set env var: $env:HF_TOKEN = "hf_...")
#   3. The 50% shard files must exist (run Phase 1 build scripts first)
#
# After the run completes (~2-4 hrs for 57 tasks), retrieve results with:
#   modal volume get kvpress-convcodeworld-results `
#     sample_50pct_no_press_fullkv_shard0/predictions.jsonl `
#     .\predictions_sample_shard0.jsonl

param(
    [string]$ModalProfile    = "pganesh",
    [string]$GpuSpec         = "H200",
    [string]$FeedbackModel   = "google/gemma-4-26B-A4B-it",
    [string]$OutputSubdir    = "sample_50pct_no_press_fullkv_shard0",
    [string]$HfSecretName    = "hf-secret"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
$env:MSYS_NO_PATHCONV = "1"
$env:MSYS2_ARG_CONV_EXCL = "*"

$ContainerShard = "/root/kvpress/evaluation/benchmarks/convcodeworld/splits/shards/tune_50pct_seed42_shard_0_of_10.json"

# Verify the shard file exists locally before dispatching
$LocalShard = "evaluation\benchmarks\convcodeworld\splits\shards\tune_50pct_seed42_shard_0_of_10.json"
if (-not (Test-Path $LocalShard)) {
    Write-Error "Shard JSON missing: $LocalShard`nRun build_shards.py first."
    exit 1
}

Write-Host "Dispatching sample run (shard 0, 57 tasks, full KV cache)..."
Write-Host "  profile      : $ModalProfile"
Write-Host "  gpu          : $GpuSpec"
Write-Host "  output_subdir: $OutputSubdir"

$env:MODAL_PROFILE                     = $ModalProfile
$env:KV_PRESS_CONVCODEWORLD_MODAL_GPU  = $GpuSpec
$env:MODAL_HF_SECRET_NAME              = $HfSecretName
# Downgrade LF/CRLF file-modification warnings from error to warning on Windows
$env:MODAL_BUILD_VALIDATION            = "warn"

modal run -d `
  evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live `
  --benchmark-mode live `
  --model meta-llama/Meta-Llama-3.1-8B-Instruct `
  --attn-implementation flash_attention_3 `
  --feedback-model $FeedbackModel `
  --feedback-attn-implementation vllm_triton `
  --feedback-vllm-max-model-len 32768 `
  --feedback-vllm-gpu-memory-utilization 0.75 `
  --feedback-config CF_EF_UNIT_SNF `
  --press-name no_press `
  --compression-ratio 0.0 `
  --global-budget 4096 `
  --local-budget 2048 `
  --full-kv-cache `
  --code-generation-until-eos `
  --cot `
  --max-turns 10 `
  --num-eval-examples 0 `
  --task-ids "@$ContainerShard" `
  --output-subdir $OutputSubdir `
  --require-flashdecode `
  --log-level INFO

Write-Host ""
Write-Host "Run dispatched. Monitor at https://modal.com/apps"
Write-Host ""
Write-Host "When complete, retrieve predictions.jsonl:"
Write-Host "  modal volume get kvpress-convcodeworld-results ``"
Write-Host "    ${OutputSubdir}/predictions.jsonl ``"
Write-Host "    .\predictions_sample_shard0.jsonl"
