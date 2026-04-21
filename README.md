# Gauss: LLM Checkpoint Compression

Compress LLM weights using GMM + ANS.

- 4.9× compression
- bounded error: ±5e-4
- CPU-only

## Usage

```
gauss compress model.safetensors model.gauss
gauss decompress model.gauss model.safetensors
```
