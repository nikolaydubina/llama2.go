# llama2.go

Same as [llama2.c](https://github.com/karpathy/llama2.c) but in native Go.

## How to run?

1. get `tokenizer.bin` from [llama2.c](https://github.com/karpathy/llama2.c) (included)
2. get weights from from [llama2.c](https://github.com/karpathy/llama2.c). `wget https://karpathy.ai/llama2c/model44m.bin -P out44m\n./run out44m/model44m.bin`
3. `go install github.com/nikolaydubina/llama2.go`
4. `llama2.go --checkpoint out44m/model44m.bin`

## Differences from `llama2.c`

* for checkpoint not using `mmap`, instead scanning file

## References

* https://github.com/karpathy/llama2.c
* https://github.com/poudels14/llama2_rs
