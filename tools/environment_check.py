import os, platform
report = {}

try:
    import torch
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        report["device"] = dev
        report["cc"] = f"{props.major}.{props.minor}"
        report["vram_gb"] = round(props.total_memory/1024**3, 2)
except Exception as e:
    report["torch_error"] = str(e)

def safe_import(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "ok")
    except Exception as e:
        return f"ERROR: {e}"

for pkg in ["diffusers","transformers","accelerate","safetensors","datasets","pillow","einops","peft","torch_fidelity"]:
    report[pkg] = safe_import(pkg)

try:
    import xformers
    report["xformers"] = getattr(xformers, "__version__", "ok")
except Exception as e:
    report["xformers"] = f"not available ({e})"

print("=== Environment Check ===")
for k,v in report.items():
    print(f"{k:16} {v}")

print("\nOS:", platform.platform())
print("PYTORCH_CUDA_ALLOC_CONF:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
