<# =====================================================================
 run_all.ps1  —  One-click pipeline for preprocess → captions → train → plots
 - Works around PowerShell backtick issues by using argument arrays.
 - Run all variants or a single one: vanilla | blip | blip_plus | all
 - Skippable stages: -DoPreprocess, -DoCaptions, -DoTrain, -DoCompare
 - If -Seed -1  → randomizes each run.
 - Auto-resume: if weights/ exists in run dir, adds --resume_from_dir.

 Requirements:
   scripts\preprocess.py
   scripts\auto_caption_blip.py
   scripts\enrich_captions.py
   scripts\train_lora.py
   scripts\plot_training.py
 ===================================================================== #>

param(
  [ValidateSet('all','vanilla','blip','blip_plus')]
  [string]$Variant = 'all',

  [switch]$DoPreprocess,
  [switch]$DoCaptions,
  [switch]$UseExistingCaptions,   # if set, skip captioning even if -DoCaptions
  [switch]$DoTrain,
  [switch]$DoCompare,

  # If omitted, we try to auto-detect python.exe (conda env, common paths, fallback "python")
  [string]$PythonPath = "",

  # Dataset + paths
  [string]$Dataset = "tattoo_v3_subset2000",
  [int]$Size = 384,                       # preprocess size
  [string]$RawRoot = "data\raw",
  [string]$ProcessedRoot = "data\processed",

  # Training hyperparams (defaults safe for 8–12 GB GPUs; adjust as needed)
  [int]$Rank = 8,
  [int]$Alpha = 8,
  [int]$Res = 384,
  [int]$MaxSteps = 1024,
  [int]$BatchSize = 8,
  [int]$GradAccum = 16,
  [double]$ValSplit = 0.05,
  [int]$EvalEvery = 64,
  [int]$EvalStepsInfer = 16,
  [double]$EvalGuidance = 6.0,

  # Seed: pass -1 to randomize each run
  [int]$Seed = 32
)

# ---------------------------
# Helper functions
# ---------------------------
function Get-PythonPath {
  param([string]$Preferred)
  if ($Preferred -and (Test-Path $Preferred)) { return $Preferred }

  if ($env:CONDA_PREFIX) {
    $condaPy = Join-Path $env:CONDA_PREFIX "python.exe"
    if (Test-Path $condaPy) { return $condaPy }
  }

  $common = @(
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python39\python.exe",
    "C:\Users\Omer\anaconda3\envs\tattoo_lora\python.exe"
  )
  foreach ($c in $common) { if (Test-Path $c) { return $c } }

  return "python"  # last resort: from PATH
}

function Assert-File {
  param([string]$Path, [string]$Friendly)
  if (-not (Test-Path $Path)) {
    throw "Missing required $Friendly file: $Path"
  }
}

function Run-Py {
  param(
    [string]$PythonExe,
    [string[]]$ArgsArray
  )
  Write-Host ">>> $PythonExe $($ArgsArray -join ' ')" -ForegroundColor Cyan
  & $PythonExe $ArgsArray
  if ($LASTEXITCODE -ne 0) {
    throw "Process failed (exit $LASTEXITCODE): $($ArgsArray -join ' ')"
  }
}

function Get-Seed {
  param([int]$Seed)
  if ($Seed -ge 0) { return $Seed }
  # randomize (0..1e6)
  return (Get-Random -Minimum 0 -Maximum 1000000)
}

function Get-LogCsv {
  param([string]$RunName)
  $p1 = Join-Path "runs\logs" "$RunName\metrics.csv"
  $p2 = Join-Path "runs\logs" "$RunName.csv"
  if (Test-Path $p1) { return $p1 }
  if (Test-Path $p2) { return $p2 }
  return $null
}

# ---------------------------
# Resolve paths & scripts
# ---------------------------
$py = Get-PythonPath -Preferred $PythonPath

# Try printing python version (non-fatal)
try { & $py -V | Out-Host } catch { Write-Warning "Could not query Python version from: $py" }

$procDir = Join-Path $ProcessedRoot $Dataset
$imgDir  = Join-Path $procDir "images"

$capVanillaDir  = $imgDir
$capBlipDir     = Join-Path $procDir "captions_blip"
$capBlipPlusDir = Join-Path $procDir "captions_blip_plus"

$runsRoot = "runs\lora"
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null
New-Item -ItemType Directory -Force -Path "runs\logs" | Out-Null

# Scripts (fail early if missing)
Assert-File -Path "scripts\train_lora.py" -Friendly "training"
Assert-File -Path "scripts\plot_training.py" -Friendly "plotting"
if ($DoPreprocess) { Assert-File -Path "scripts\preprocess.py" -Friendly "preprocess" }
if ($DoCaptions -and -not $UseExistingCaptions) {
  Assert-File -Path "scripts\auto_caption_blip.py" -Friendly "BLIP captioning"
  Assert-File -Path "scripts\enrich_captions.py" -Friendly "BLIP+ captioning"
}

# ---------------------------
# Print settings
# ---------------------------
$seedToUse = Get-Seed -Seed $Seed

Write-Host "=== SETTINGS ==="
Write-Host ("Variant:            {0}" -f $Variant)
Write-Host ("Stages:             Preprocess={0}  Captions={1} (UseExisting={2})  Train={3}  Compare={4}" -f [bool]$DoPreprocess, [bool]$DoCaptions, [bool]$UseExistingCaptions, [bool]$DoTrain, [bool]$DoCompare)
Write-Host ("Dataset:            {0}" -f $Dataset)
Write-Host ("Images dir:         {0}" -f $imgDir)
Write-Host ("Caption dirs:       BLIP={0} | BLIP+={1} | Vanilla={2}" -f $capBlipDir, $capBlipPlusDir, $capVanillaDir)
Write-Host ("Training:           steps={0}, bs={1}, accum={2}, res={3}, rank={4}, alpha={5}, seed={6}" -f $MaxSteps, $BatchSize, $GradAccum, $Res, $Rank, $Alpha, $seedToUse)
Write-Host "=================="

# ---------------------------
# (1) Preprocess
# ---------------------------
if ($DoPreprocess) {
  Write-Host "=== (1) Preprocess images ===" -ForegroundColor Yellow
  $preArgs = @(
    "scripts\preprocess.py",
    "--dataset_name", $Dataset,
    "--input_dir", $RawRoot,
    "--size", $Size.ToString(),
    "--out_root", $ProcessedRoot,
    "--images_only"
  )
  Run-Py -PythonExe $py -ArgsArray $preArgs
} else {
  Write-Host "=== (1) Preprocess skipped ==="
}

# sanity: images must exist
if (-not (Test-Path $imgDir)) { throw "Images folder not found: $imgDir" }
$imgCount = (Get-ChildItem $imgDir -Filter *.png -File -ErrorAction SilentlyContinue).Count
if ($imgCount -le 0) {
  # allow jpg/jpeg/webp too
  $imgCount = (Get-ChildItem $imgDir -Include *.png,*.jpg,*.jpeg,*.webp -File -ErrorAction SilentlyContinue).Count
}
if ($imgCount -le 0) { throw "No images found in $imgDir" }

# ---------------------------
# (2) Captions
# ---------------------------
$willCaption = $DoCaptions -and (-not $UseExistingCaptions) -and ($Variant -ne 'vanilla') # vanilla doesn't need txt
if ($willCaption) {
  Write-Host "=== (2) Captioning ===" -ForegroundColor Yellow

  if ($Variant -eq 'all' -or $Variant -eq 'blip') {
    New-Item -ItemType Directory -Force -Path $capBlipDir | Out-Null
    $blipArgs = @(
      "scripts\auto_caption_blip.py",
      "--img_dir", $imgDir,
      "--out_dir", $capBlipDir,
      "--model", "Salesforce/blip-image-captioning-base",
      "--batch_size", "2",
      "--max_new_tokens", "32"
    )
    Run-Py -PythonExe $py -ArgsArray $blipArgs
  }

  if ($Variant -eq 'all' -or $Variant -eq 'blip_plus') {
    New-Item -ItemType Directory -Force -Path $capBlipPlusDir | Out-Null
    $blipPlusArgs = @(
      "scripts\enrich_captions.py",
      "--img_dir", $imgDir,
      "--out_dir", $capBlipPlusDir,
      "--append_style", ", clean line tattoo, high contrast, stencil, no shading",
      "--batch_size", "2",
      "--model", "Salesforce/blip-image-captioning-base"
    )
    Run-Py -PythonExe $py -ArgsArray $blipPlusArgs
  }

  Write-Host "=== Caption counts ==="
  Write-Host ("BLIP     : {0}" -f (Get-ChildItem $capBlipDir -Filter *.txt -File -ErrorAction SilentlyContinue).Count)
  Write-Host ("BLIP+    : {0}" -f (Get-ChildItem $capBlipPlusDir -Filter *.txt -File -ErrorAction SilentlyContinue).Count)
  Write-Host ("Images   : {0}" -f $imgCount)
} else {
  Write-Host "=== (2) Captioning skipped ==="
}

# ---------------------------
# (3) Train
# ---------------------------
function New-TrainArgs {
  param(
    [string]$DataImages,
    [string]$DataCaptions,
    [string]$OutputDir,
    [int]$SeedLocal
  )
  $args = @(
    "scripts\train_lora.py",
    "--data_images", $DataImages,
    "--data_captions", $DataCaptions,
    "--output_dir", $OutputDir,
    "--rank", $Rank.ToString(), "--alpha", $Alpha.ToString(),
    "--resolution", $Res.ToString(),
    "--max_steps", $MaxSteps.ToString(),
    "--batch_size", $BatchSize.ToString(),
    "--grad_accum", $GradAccum.ToString(),
    "--val_split", $ValSplit.ToString(),
    "--save_every", "0",
    "--eval_every", $EvalEvery.ToString(),
    "--eval_steps_infer", $EvalStepsInfer.ToString(),
    "--eval_guidance", $EvalGuidance.ToString(),
    "--eval_width", $Res.ToString(), "--eval_height", $Res.ToString(),
    "--seed", $SeedLocal.ToString(),
    "--plot_at_end", "--debug_samples", "5"
  )

  $weightsDir = Join-Path $OutputDir "weights"
  if (Test-Path $weightsDir) {
    $args += @("--resume_from_dir", $weightsDir)
  }
  return $args
}

$runVanilla = "${Dataset}_vanilla_r${Rank}a${Alpha}_res${Res}_s${MaxSteps}"
$runBlip    = "${Dataset}_blip_r${Rank}a${Alpha}_res${Res}_s${MaxSteps}"
$runBlpPlus = "${Dataset}_blip_plus_r${Rank}a${Alpha}_res${Res}_s${MaxSteps}"

# Decide which runs to execute
$runs = @()
switch ($Variant) {
  'vanilla'   { $runs = @(@{name=$runVanilla; caps=$capVanillaDir}) }
  'blip'      { $runs = @(@{name=$runBlip;    caps=$capBlipDir}) }
  'blip_plus' { $runs = @(@{name=$runBlpPlus; caps=$capBlipPlusDir}) }
  'all'       {
    $runs = @(
      @{name=$runVanilla; caps=$capVanillaDir},
      @{name=$runBlip;    caps=$capBlipDir},
      @{name=$runBlpPlus; caps=$capBlipPlusDir}
    )
  }
}

$completedRuns = @()

if ($DoTrain) {
  Write-Host "=== (3) Training ===" -ForegroundColor Yellow

  foreach ($r in $runs) {
    $outDir = Join-Path $runsRoot $r.name
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $seedThisRun = Get-Seed -Seed $Seed
    Write-Host ("-> {0} (seed={1})" -f $r.name, $seedThisRun) -ForegroundColor Green

    $targs = New-TrainArgs -DataImages $imgDir -DataCaptions $r.caps -OutputDir $outDir -SeedLocal $seedThisRun
    Run-Py -PythonExe $py -ArgsArray $targs

    $completedRuns += $r.name
  }
} else {
  Write-Host "=== (3) Training skipped ==="
}

# ---------------------------
# (4) Compare plots (optional)
# ---------------------------
if ($DoCompare) {
  Write-Host "=== (4) Compare loss curves ===" -ForegroundColor Yellow
  $csvs = @()
  $labels = @()

  foreach ($r in $runs) {
    $csv = Get-LogCsv -RunName $r.name
    if ($csv) {
      $csvs += $csv
      if ($r.name -like "*vanilla*") { $labels += "vanilla" }
      elseif ($r.name -like "*blip_plus*") { $labels += "blip_plus" }
      else { $labels += "blip" }
    } else {
      Write-Warning "No log CSV found for $($r.name) (looked for runs\logs\$($r.name)\metrics.csv and runs\logs\$($r.name).csv)"
    }
  }

  if ($csvs.Count -ge 2) {
    $outCompare = "runs\logs\compare_$($Variant)_$(Get-Date -Format 'yyyyMMdd_HHmmss').png"
    $plotArgs = @("scripts\plot_training.py", "--csv") + $csvs + @("--labels") + $labels + @("--out", $outCompare)
    Run-Py -PythonExe $py -ArgsArray $plotArgs
    Write-Host "Comparison plot: $outCompare"
  } else {
    Write-Host "Not enough CSVs to compare (need at least 2)."
  }
} else {
  Write-Host "=== (4) Compare skipped ==="
}

Write-Host "=== DONE ==="
if ($completedRuns.Count -gt 0) {
  Write-Host "Completed runs:"
  $completedRuns | ForEach-Object { Write-Host "  $_" }
}
