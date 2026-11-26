# GPU matrix

Boot-parity runs of the page across WebGPU adapter classes. One Playwright
project per class; the spec asserts the adapter that answered is the class
the project pins, then checks `__PARITY_REPORT__` (all clean legs pass, all
fault legs detected).

Runs on demand with no CI workflow: `workers: 1`, serial adapter-per-host.

## Matrix

| project         | GPU class                          | driver select                                |
|-----------------|------------------------------------|----------------------------------------------|
| `swiftshader`   | CPU (Dawn SwiftShader)             | `--use-webgpu-adapter=swiftshader`           |
| `apple-metal-3` | Apple Silicon, Metal-3             | `--use-webgpu-adapter=metal`                 |
| `intel-anv`     | Intel Arc (ANV, Vulkan)            | Vulkan adapter + `VK_ICD_FILENAMES` -> ANV   |
| `nvidia-vulkan` | NVIDIA discrete, proprietary Vulkan | Vulkan adapter + `VK_ICD_FILENAMES` -> NVIDIA |

To add or swap a class, edit `GPU_MATRIX` in `tests/gpu-matrix/matrix.ts`
only - the config and spec both key off the project name. A commented
`amd-radv` row is staged as an example.

## Run

```sh
npm install          # picks up @playwright/test + browsers below
npx playwright install chromium

npm run test:gpu -- --project=swiftshader            # baseline, runs anywhere
```

### Real-GPU classes

Two modes per project, selected by `GPU_CDP_<PROJECT>`.

Remote (recommended when the test host has no GPU): serve the page where the
GPU host can reach it, run Chromium there with CDP open, and point
Playwright at the socket.

```sh
# here: serve the built page on a reachable interface (webServer is skipped
# while GPU_BASE_URL is set)
npm run build && npm run preview -- --port 4173 --strictPort --host 0.0.0.0

# on the GPU host (CDP binds loopback by default; --remote-debugging-address
# opens it on all interfaces, or tunnel the port instead). Swap
# --use-webgpu-adapter=vulkan for swiftshader to smoke the path on a
# GPU-less host - that class also needs --enable-unsafe-swiftshader.
chromium --headless=new \
  --remote-debugging-port=9222 --remote-debugging-address=0.0.0.0 \
  --enable-unsafe-webgpu --enable-features=Vulkan \
  --use-webgpu-adapter=vulkan --ignore-gpu-blocklist about:blank

# here
GPU_BASE_URL=http://<this-host>:4173 \
GPU_CDP_INTEL_ANV=http://<gpu-host>:9222 \
  npx playwright test --project=intel-anv
```

`GPU_CDP_<NAME>` takes an http:// or ws:// endpoint (the fixture uses
`connectOverCDP`, not the Playwright server protocol). The browser on the
GPU host resolves `GPU_BASE_URL`, so it must name an address that host can
route to - not loopback.

Local (when the config runs on the GPU host itself): the fixture launches
Chromium with the project's `launchArgs`. Pin the driver:

```sh
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/<anv-icd>.json \
  npx playwright test --project=intel-anv
```

The runner inherits `process.env`, so `VK_ICD_FILENAMES` (or
`VK_DRIVER_FILES` on newer Mesa) set on the runner reaches the launched
browser. If a host exposes more than one Vulkan device, the ICD file is the
discriminator.

### Model-load smoke (optional)

`GPU_MATRIX_GENERATE=1` adds a second test per project: a real
`__SMOKE_GENERATE__` call (320 MB model fetch plus a 24-token decode) on the
same adapter. Heavy and opt-in only.

## Known gotchas

- Container Chromium may need `--no-sandbox` / `--disable-gpu-sandbox` for
  real GPU access; add to the project's `launchArgs` if the run on the GPU
  host fails to enumerate an adapter. Software-only classes additionally
  need `--enable-unsafe-swiftshader` (Chrome 120+ gate) - Playwright's own
  launcher adds it, a manual CDP browser launch does not.
- `adapter.info` strings are browser-version dependent (`vendor` can be
  empty on some builds). The spec matches a joined vendor + architecture +
  device + description string, but a new class may need its regex widened.
- Fault legs assert `!passed` by design - a "fault detected" green means the
  page reported the injected failure as a parity failure.
