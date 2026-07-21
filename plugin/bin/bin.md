# Binaries for /bin

## Required binaries

| Binary Name | Description |
|---|---|
| `cpuKM.exe` | K‑means in CPU |
| `gpuKM.exe` | K‑means in GPU |
| `cpuGMM.exe` | GMM in CPU |
| `gpuGMM.exe` | GMM in GPU |

## Steps to insert

1. **Compile the sources** (if not already done) and locate the resulting executables.  

2. **Create the `/bin` folder** (only if it does not exist).

3. **Copy the binaries into the `/bin` folder.**

4. Ensure they are **executable** with: ```chmod +x /bin/cpuKM.exe /bin/gpuKM.exe /bin/cpuGMM.exe /bin/gpuGMM.exe```.