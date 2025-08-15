# CK gemm a4w4 blockscale tune

1. Install aiter:
`cd $aiter_path`
`python3 setup.py develop`

2. Add GEMM shapes in `aiter/configs/a4w4_blockscale_untuned_gemm.csv`
    |**M**|**N**|**K**|
    |-----|-----|-----|
    |128  |1536 |7168 |

3. Start tuning:
Run the following cmd to start tuning, please wait a few minutes as it will build gemm_a4w4_blockscale_tune via jit:
`GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/work/llvm-project/build/bin/ python3 csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py -i aiter/configs/a4w4_blockscale_untuned_gemm.csv -o aiter/configs/a4w4_blockscale_tuned_gemm.csv`
You can find the results of the tuning in `aiter/configs/a4w4_blockscale_tuned_gemm.csv`.
    |**cu_num**|**M**|**N**|**K**|**kernelId**|**splitK**|**us**|**kernelName**|
    |----------|-----|-----|-----|------------|----------|------|--------------|
    |80        |128  |1536 |7168 |23          |0         |32.99 |xxxxxxxx      |

    `cu_num` means the number of compute units, and it is used to distinguish between graphics.

4. Build tuned kernels and test:
Test the performance, modify the test instance in `op_tests/test_gemm_a4w4_blockscale.py` and run it, please wait a few minutes as it will build gemm_a4w4_blockscale tuned kernels in `aiter/configs/a4w4_blockscale_tuned_gemm.csv` via jit:
`GEMM_A4W4_BLOCKWISE_HIP_CLANG_PATH=/work/llvm-project/build/bin/ python3 op_tests/test_gemm_a4w4_blockscale.py`
If you have built gemm_a4w4 kernels brefore tuning new GEMM shapes, please add `AITER_REBUILD=1` before your test cmd, such as `AITER_REBUILD=1 python3 op_tests/test_gemm_a4w4_blockscale.py`. It will rebuild kernels from `aiter/configs/a4w4_blockscale_tuned_gemm.csv`.

## More
If you use flag `PREBUILD_KERNELS=1` when you install aiter, it will build gemm a4w4 kernels in tuned gemm csv by default. If you want to use the new result of gemm_a4w4_tune, please remove `build` and `*.so` in `aiter/jit` first, then re-intall aiter after finishing tune. This can take a lot of time and is not recommended.
