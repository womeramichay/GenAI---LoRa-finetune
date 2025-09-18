# =========================
# run_all.ps1  (copy/paste this whole block to a .ps1 file)
# =========================

# ---- Settings ----
$py = "C:\Users\Omer\anaconda3\envs\tattoo_lora\python.exe"   # path to your env's python
$dataset = "tattoo_v3_subset2000"
$size = 384

# Paths
$rawDir  = "data\raw"
$procDir = "data\processed\$dataset"
$imgDir  = "$procDir\images"

$capVanillaDir   = $imgDir                            # vanilla = captions next to images (fallback)
$capBlipDir      = "$procDir\captions_blip"
$capBlipPlusDir  = "$procDir\captions_blip_plus"

# Training common
$rank = 8
$alpha = 8
$res = 384
$maxSteps = 512
$batchSize = 4
$gradAccum = 8
$valSplit = 0.05
$evalEvery = 100
$seed = 11

# Output runs
$runsRoot = "runs\lora"
$runVanilla   = "${dataset}_vanilla_r${rank}a${alpha}_res${res}_s${maxSteps}"
$runBlip      = "${dataset}_blip_r${rank}a${alpha}_res${res}_s${maxSteps}"
$runBlipPlus  = "${dataset}_blip_plus_r${rank}a${alpha}_res${res}_s${maxSteps}"

# ----------------------------
Write-Host "=== (1) Preprocess images (if needed) ==="
& $py "scripts\preprocess.py" --dataset_name $dataset --input_dir $rawDir --size $size --out_root "data\processed" --images_only

Write-Host "=== (2) Create captions: BLIP and BLIP+ ==="
# BLIP captions
New-Item -ItemType Directory -Force -Path $capBlipDir | Out-Null
& $py "scripts\auto_caption_blip.py" --img_dir $imgDir --out_dir $capBlipDir --model "Salesforce/blip-image-captioning-base" --batch_size 2 --max_new_tokens 32

# BLIP+ captions (richer / style-appended)
New-Item -ItemType Directory -Force -Path $capBlipPlusDir | Out-Null
& $py "scripts\enrich_captions.py" --img_dir $imgDir --out_dir $capBlipPlusDir --append_style ", clean line tattoo, high contrast, stencil, no shading" --batch_size 2 --model "Salesforce/blip-image-captioning-base"

# Quick sanity counts
Write-Host "=== Caption counts ==="
Write-Host ("BLIP     :", (Get-ChildItem $capBlipDir -Filter *.txt).Count)
Write-Host ("BLIP+    :", (Get-ChildItem $capBlipPlusDir -Filter *.txt).Count)
Write-Host ("Images   :", (Get-ChildItem $imgDir -Filter *.png).Count)

# ----------------------------
Write-Host "=== (3) Train: VANILLA (uses fallback caption if no .txt exists) ==="
& $py "scripts\train_lora.py" `
  --data_images  $imgDir `
  --data_captions $capVanillaDir `
  --output_dir   (Join-Path $runsRoot $runVanilla) `
  --rank $rank --alpha $alpha --resolution $res `
  --max_steps $maxSteps --batch_size $batchSize --grad_accum $gradAccum `
  --val_split $valSplit --save_every 0 --eval_every $evalEvery `
  --eval_steps_infer 16 --eval_guidance 6.0 --eval_width $res --eval_height $res `
  --seed $seed --plot_at_end --debug_samples 5

Write-Host "=== (4) Train: BLIP captions ==="
& $py "scripts\train_lora.py" `
  --data_images  $imgDir `
  --data_captions $capBlipDir `
  --output_dir   (Join-Path $runsRoot $runBlip) `
  --rank $rank --alpha $alpha --resolution $res `
  --max_steps $maxSteps --batch_size $batchSize --grad_accum $gradAccum `
  --val_split $valSplit --save_every 0 --eval_every $evalEvery `
  --eval_steps_infer 16 --eval_guidance 6.0 --eval_width $res --eval_height $res `
  --seed $seed --plot_at_end --debug_samples 5

Write-Host "=== (5) Train: BLIP+ captions ==="
& $py "scripts\train_lora.py" `
  --data_images  $imgDir `
  --data_captions $capBlipPlusDir `
  --output_dir   (Join-Path $runsRoot $runBlipPlus) `
  --rank $rank --alpha $alpha --resolution $res `
  --max_steps $maxSteps --batch_size $batchSize --grad_accum $gradAccum `
  --val_split $valSplit --save_every 0 --eval_every $evalEvery `
  --eval_steps_infer 16 --eval_guidance 6.0 --eval_width $res --eval_height $res `
  --seed $seed --plot_at_end --debug_samples 5

# ----------------------------
Write-Host "=== (6) Compare loss curves across runs ==="
$log1 = "runs\logs\$runVanilla\metrics.csv"
$log2 = "runs\logs\$runBlip\metrics.csv"
$log3 = "runs\logs\$runBlipPlus\metrics.csv"
$outCompare = "runs\logs\compare_vanilla_blip_blipplus.png"

& $py "scripts\plot_training.py" --csv $log1 $log2 $log3 --labels vanilla blip blip_plus --out $outCompare

Write-Host "=== DONE ==="
Write-Host "Per-run plots:"
Write-Host "  runs\logs\$runVanilla\loss.png"
Write-Host "  runs\logs\$runBlip\loss.png"
Write-Host "  runs\logs\$runBlipPlus\loss.png"
Write-Host "Comparison plot:"
Write-Host "  $outCompare"
