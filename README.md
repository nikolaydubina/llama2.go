## llama2.go

Native Go version of [llama2.c](https://github.com/karpathy/llama2.c).

## Wait what?

It is pure Go inference code ported from experimental implementation by [Andrej Karpathy](https://en.wikipedia.org/wiki/Andrej_Karpathy) of latest as of `2023-07-25` LLM model from Meta [LLAMA-2](https://ai.meta.com/llama/).  

## How to run?

1. get `tokenizer.bin` from [llama2.c](https://github.com/karpathy/llama2.c) (included)
2. get weights from from [llama2.c](https://github.com/karpathy/llama2.c). `wget https://karpathy.ai/llama2c/model44m.bin -P out44m`
3. `go install github.com/nikolaydubina/llama2.go`
4. `llama2.go --checkpoint out44m/model44m.bin`

Example output:
```

```

## Differences from `llama2.c`

* for checkpoint not using `mmap`, instead scanning file

## Performance



```
llama.c
achieved tok/s: 125.860374
```

## References

* https://github.com/karpathy/llama2.c
* https://github.com/poudels14/llama2_rs
