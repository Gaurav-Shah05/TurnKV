# Dispatch all 10 shards for the full 50% no_press / full-kv-cache run.
# Run from kvpress/:  .\dispatch_full_50pct.ps1
param(
    [string]$ConfigLabel = "no_press_global4096_local2048_compressratio0.0_split570",
    [string]$ModalProfile = "pganesh",
    [string]$GpuSpec = "H200",
    [string]$HfSecretName = "hf-secret",
    [int]$NumShards = 10
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
$env:MSYS_NO_PATHCONV = "1"
$env:MSYS2_ARG_CONV_EXCL = "*"
$env:MODAL_PROFILE    = $ModalProfile
$env:KV_PRESS_CONVCODEWORLD_MODAL_GPU = $GpuSpec
$env:MODAL_HF_SECRET_NAME = $HfSecretName
$env:MODAL_BUILD_VALIDATION = "warn"

$RunTs = Get-Date -Format "yyyyMMdd_HHmmss"
$RunTag = "no_press_${ConfigLabel}_live_smoke"
$LogDir = "..\..\..\..\.modal_diag\${RunTag}_${RunTs}"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$IndexFile = "$LogDir\index.txt"
"# full 50pct no_press run label=$ConfigLabel profile=$ModalProfile gpu=$GpuSpec ts=$RunTs" | Out-File $IndexFile

$ContainerShardDir = "/root/kvpress/evaluation/benchmarks/convcodeworld/splits/shards"
$OutputTag = "${RunTag}_${RunTs}"

Write-Host "Dispatching $NumShards shards for label: $ConfigLabel"
Write-Host "Log dir: $LogDir"

for ($shard = 0; $shard -lt $NumShards; $shard++) {
    $ShardFile = "$ContainerShardDir/tune_50pct_seed42_shard_${shard}_of_${NumShards}.json"
    $OutputSubdir = "${OutputTag}/shard_${shard}_of_${NumShards}"
    $LogFile = "$LogDir\shard${shard}.log"

    Write-Host "[shard $shard/$($NumShards-1)] dispatching -> $LogFile"

    Start-Process modal -ArgumentList @(
        "run", "-d",
        "evaluation/benchmarks/convcodeworld/modal_app.py::run_convcodeworld_live",
        "--benchmark-mode", "live",
        "--model", "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "--attn-implementation", "flash_attention_3",
        "--feedback-model", "google/gemma-4-26B-A4B-it",
        "--feedback-attn-implementation", "vllm_triton",
        "--feedback-vllm-max-model-len", "32768",
        "--feedback-vllm-gpu-memory-utilization", "0.75",
        "--feedback-config", "CF_EF_UNIT_SNF",
        "--press-name", "no_press",
        "--compression-ratio", "0.0",
        "--global-budget", "4096",
        "--local-budget", "2048",
        "--full-kv-cache",
        "--code-generation-until-eos",
        "--cot",
        "--max-turns", "10",
        "--num-eval-examples", "0",
        "--task-ids", "@$ShardFile",
        "--output-subdir", $OutputSubdir,
        "--require-flashdecode",
        "--log-level", "INFO"
    ) -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err" -NoNewWindow

    "shard=$shard json=$ShardFile log=$LogFile output_subdir=$OutputSubdir" | Out-File $IndexFile -Append
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "All $NumShards shards dispatched."
Write-Host "Index: $IndexFile"
Write-Host ""
Write-Host "Monitor on Modal dashboard: https://modal.com/apps"
Write-Host ""
Write-Host "After all shards complete, retrieve results from:"
Write-Host "  modal volume ls kvpress-convcodeworld-results"
