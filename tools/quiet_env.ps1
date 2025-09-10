# tools\quiet_env.ps1
$env:PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
$env:HF_HUB_DISABLE_PROGRESS_BARS="1"
$env:TRANSFORMERS_VERBOSITY="error"
$env:DIFFUSERS_VERBOSITY="error"
# If everything is fully cached, uncomment the next line to kill all HF networking chatter
# $env:HF_HUB_OFFLINE="1"
Write-Host "Quiet env set."
