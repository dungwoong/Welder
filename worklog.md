# 2025/06/25
- Look at attention config
- next steps are to try feeding in a different attn config(try eg. switching the stuff around, and then removing numerical stability entirely) to see if welder picks up

# 2025/06/24
```bash
python3 ./testing/torch2onnx.py gemm --bs 4096 --fp16 --prefix workdir
python3 ./testing/relay_test.py workdir
python3 ./testing/test_welder_perf.py workdir
```

- Got a gemm onnx model by modifying `testing/model/pytorch`
- Ran `relay_test` to test the compiler
- it split the thing into a dotsplitk and a sum_1. Note that mnk=1024
- you can find comments in dotsplitk. It doesn't use tensor cores. Let's try to make it use tensor cores

### Using Tensor Core Policy
- In `base_tunner.py`: if node.getTag("tensorCoreConfig") then it has TCPolicy
- in `engine.py`, we tune the new group of nodes
- I think in `relay_test` it runs `mod = welder.relay.transform.AnnotateTensorCore()(mod)`, need to edit that
- ok it's because I set batch size to 1 in torch2onnx so it was doing 1x1024x1024x1024. Should do bs 1024.
- CHECK tensorCorePolicy by printing node._tag
- On 4096 they claim 130TFLOPs.
- actually if you run `python3 ./testing/test_welder_perf.py workdir` you get 121.6 TFLOPs, so a lil under cublas
- looked at it a bit, it's hard to completely reverse-engineer so maybe we can look at higher level IR

### Adding Attn Graph
- Added in `model/pytorch/kevin.py` and tested against torch
- Ran it through welder, it claims 315 TFLOPs but it's half precision, so I should go see what I can do there.
- Tomorrow I should look at intermediate rep and figure out what's going on.

### Exposing intermediate representations
- I have a writeup in notes.md about how I did it with GEMM. I'll try FlashAttention after.
