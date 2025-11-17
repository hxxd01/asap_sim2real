# Atom SDK

This SDK provides both high-level RPC and low-level DDS interfaces for robot control.

## Build Environment
- **OS:** Ubuntu 20.04 LTS
- **Compiler:** GCC 9.4.0

## How to Build

```bash
mkdir build
cd build
cmake ..
make
```

## How to Run Examples

> **Warning:** The robot will move when running these examples. Please ensure your environment is safe before proceeding.

### 1. Run the high-level RPC example

```bash
cd build
./rpc_test
```

### 2. Run the low-level DDS example

First, ensure the robot is switched to "Low-Level Control" mode.

```bash
cd build
./bridge_test
```
