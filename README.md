# Simulator for AttAcc
This repository includes Python-based simulator designed to analyze the transformer-based generation model (TbGM) inference in a heterogeneous system consisting of an xPU and an Attention Accelerator (AttAcc). 
AttAcc is an accelerator for the attention layer of TbGM, which consists of an HBM-based processing-in-memory (PIM) structure.
In simulating an xPU and AttAcc system, the simulator outputs the performance and energy usage of the xPU, while the behavior of AttAcc is simulated using a properly modified [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2).
We set the memory device of AttAcc in Ramulator2 to HBM3 and implemented AttAcc\_bank, AttAcc\_BG, and AttAcc\_buffer, which represent AttAcc deploying processing units per bank, per bank group, or per pseudo-channel (on the buffer die), respectively.
For more details of AttAcc, please check the [paper](https://dl.acm.org/doi/10.1145/3620665.3640422) **AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference** published at [ASPLOS 2024](https://www.asplos-conference.org/asplos2024).

 
## Prerequisites
- Python
- cmake, g++, and clang++ (for building Ramulator2)

AttAcc simulator is tested under the following system.

* OS: Ubuntu 22.04.3 LTS (Kernel 6.1.45)
* Compiler: g++ version 12.3.0
* python 3.8.8

We use a similar build system (CMake) as original Ramulator 2.0, which automatically downloads following external libraries.
- [argparse](https://github.com/p-ranav/argparse)
- [spdlog](https://github.com/gabime/spdlog)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)


## How to install
1. Clone the Github repository

```bash
$ git clone https://github.com/scale-snu/attacc_simulator.git
$ cd attacc_simulator
$ git submodule update --init --recursive
``` 

2. Build Ramulator2
```bash
$ bash set_pim_ramulator.sh 
$ cd ramulator2
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp ramulator2 ../ramulator2
$ cd ../../
```

## How to run

### Run GPU simulator 
```bash
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ python main.py --system {} --gpu {} --ngpu {} --model {} --lin {} --lout {} --batch {} --pim {} --powerlimit --ffopt --pipeopt

$ python main.py --help

    ## set system configuration
    parser.add_argument("--system",  type=str, default="dgx",
            help="dgx(each GPU has 80GB HBM), \
                  dgx-cpu (In dgx-base, offloading the attention layer to cpu), \
                  dgx-attacc (dgx-base + attacc")
    parser.add_argument("--gpu", type=str, default='A100a', 
            help="GPU type (A100a, A100, and H100), A100a is A100 with HBM3")
    parser.add_argument("--ngpu", type=int, default=8, 
            help="number of GPUs")
    parser.add_argument("--gmemcap",
                        type=int,
                        default=80,
                        help="memory capacity per GPU (GB).  default=80")



    ## set attacc configuration
    parser.add_argument("--pim", type=str, default='bank',
            help="pim mode. list: bank, bg, buffer")
    parser.add_argument("--powerlimit",  action='store_true', 
            help="power constraint for PIM ")
    parser.add_argument("--ffopt",  action='store_true', 
            help="apply feedforward parallel optimization ")
    parser.add_argument("--pipeopt",  action='store_true', 
            help="apply pipeline optimization ")


    ## set model and service environment
    parser.add_argument("--model", type=str, default='GPT-175B', 
            help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word", type=int, default='2', 
            help="word size (precision): 1(INT8), 2(FP16)")
    parser.add_argument("--lin",  type=int, default=2048,
            help="input sequence length")
    parser.add_argument("--lout",  type=int, default=128,
            help="number of generated tokens")
    parser.add_argument("--batch", type=int, default=1,
            help="batch size, default = 1")
```

### Examples
```bash 
# dgx (A100 with HBM3) example 
$ python main.py --system dgx --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1

# 2xdgx (A100 with HBM3) example 
$ python main.py --system dgx --gpu A100a --ngpu 16 --model GPT-175B --lin 2048 --lout 128 --batch 1

# dgx-attacc (based HBM3) example 
 ## bank level PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bank --powerlimit --ffopt --pipeopt

 ## bank group level PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim bg --powerlimit --ffopt --pipeopt

 ## buffer level PIM
$ python main.py --system dgx-attacc --gpu A100a --ngpu 8 --model GPT-175B --lin 2048 --lout 128 --batch 1 --pim buffer --powerlimit --ffopt --pipeopt 

```

## Details of the Ramulator for AttAcc
### How to Run
1. Generate PIM command traces for the Transformer-based Generative Model.
```bash
$ cd ramulator2
$ cd trace_gen
$ python gen_trace_attacc_bank.py
$ python gen_trace_attacc_bg.py
$ python gen_trace_attacc_buffer.py
```

This produces `attacc_bank.trace`, `attacc_bg.trace`, and `attacc_buffer.trace` which are GPT-175B traces of attention layer in a single decoder for AttAcc\_bank, AttAcc\_BG, AttAcc\_buffer, respectively.


You can change the model, batch, and request configuration by setting arguments as below.
```python
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=1, 
                      help="Number of heads, default=1")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace", 
                      help="output path")
```

2. Run Ramulator-AttAcc
```bash
$ ./ramulator2 -f attacc_bank.yaml
$ ./ramulator2 -f attacc_bg.yaml
$ ./ramulator2 -f attacc_buffer.yaml
```

This will print the total number of DRAM/PIM request and total elapsed memory cycles (`memory_system_cycles`).

The command log will be generated in `log` directory.


### Modeling AttAcc with a Power Contraint
We reflect the DRAM power constraint to AttAcc by increasing the delay between consecutive MAC commands (`nCCDAB`, `nCCDSB`).

We calculate these delay with the activation and read energy.

To evaulate AttAcc with no power constraint (NPC), uncomment `preset: HBM3_5.2Gbps_NPC` and comment out `preset: HBM3_5.2Gbps` in yaml config files.




## Contact
Jaehyun Park jhpark@scale.snu.ac.kr

Jaewan Choi jwchoi@scale.snu.ac.kr
