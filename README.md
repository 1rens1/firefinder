# Firefinder

Python project managed with `uv` and one root `.venv`.

This project targets a Raspberry Pi 5 with the Raspberry Pi AI Camera module
(Sony IMX500). The camera stack is installed from Raspberry Pi OS packages via
`imx500-all`; `picamera2`, OpenCV, libcamera, firmware, models, and native
bindings should not be installed from PyPI.

## Setup

On the Raspberry Pi:

```sh
./scripts/setup-pi.sh
```

The setup script installs `imx500-all`, creates `.venv` with access to system
site packages, syncs the project, and verifies the AI Camera imports.

### I2C Setup

Enable I2C for the MLX90640.

1. `sudo raspi-config`
2. `Interface Options > I2C > Enable`
3. `sudo reboot`

## Structure

```text
src/
  cam_demo/
  firefinder/
  mlx_demo/
```

## Run

```sh
uv run firefinder-camera-demo
uv run firefinder-object-detection-demo
```

