# Cool Compression (CCP)
## Philosophy and Guidelines
We exclusively accept and produce hand-crafted artisanal code made in a rural village.
We are first and foremost a C project and prioritise any C contributions completely.
Python is a painful necessity in our artisanal life.

## Install
#### clone this repo: 
`git clone https://github.com/Vescusia/cool-compression.git`

cd into it: `cd cool-compression`

#### sync: 
`uv sync --extra cpu (for CPU)`
 
`uv sync --extra cu128 (for NVIDIA)`

`uv sync --extra rocm (for ROCM)`

## Usage
### Train Model on File
`uv run --extra <DEVICE> src/ccp.py /path/to/file`

### Compress File
`uv run --extra <DEVICE> src/compressor.py /path/to/model.pt /path/to/file`

### Decompress File
`uv run --extra <DEVICE> src/pump.py /path/to/compressed.ccp`

## Contributions
When pushing to repo, play [this](https://www.youtube.com/watch?v=5nqIMN_5HIE) or it will be rejected.
You will have to attach a video to the commit.
Use CCP to compress it or it will be rejected.
