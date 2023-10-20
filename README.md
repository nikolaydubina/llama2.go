## llama2.go

<p align="center">
  <img src="assets/logo.png" width="300" height="300">
</p>

[![Go Report Card](https://goreportcard.com/badge/github.com/nikolaydubina/llama2.go)](https://goreportcard.com/report/github.com/nikolaydubina/llama2.go)
[![codecov](https://codecov.io/gh/nikolaydubina/llama2.go/branch/master/graph/badge.svg?token=OMf0git2BD)](https://codecov.io/gh/nikolaydubina/llama2.go)
[![Go Reference](https://pkg.go.dev/badge/github.com/nikolaydubina/llama2.go.svg)](https://pkg.go.dev/github.com/nikolaydubina/llama2.go)

This is a native Go inference of [LLaMA-2](https://ai.meta.com/llama/), as of `2023-08-19` state-of-the-art open source large language model from Meta. 
It is ported from [github.com/karpathy/llama2.c](https://github.com/karpathy/llama2.c)@[`bd18228`](https://github.com/karpathy/llama2.c/commit/bd182289c596fa6059eb7b3b7c8ccd04b5c90fc3) on `2023-08-19`.
Additional features may be added.

### How to run?

1. get `tokenizer.bin` from [llama2.c](https://github.com/karpathy/llama2.c)
2. get weights `wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin`
3. `go install github.com/nikolaydubina/llama2.go@latest`
4. `llama2.go -checkpoint=stories110M.bin -prompt="good morning said sun to trees"`

```bash
$ llama2.go -checkpoint=stories110M.bin -prompt="good morning said sun to trees"
2023/07/29 09:30:22 config: llama2.Config{Dim:768, HiddenDim:2048, NumLayers:12, NumHeads:12, NumKVHeads:12, VocabSize:32000, SeqLen:1024}
<s>
good morning said sun to trees: "Let's organize an operation!"
The trees clapped their branches and asked "What will we do?"
Badger smiled and replied "We will build a treehouse together!"
The trees got blocks of wood and started to build. Badger put nails in the tiny pieces of wood, while the trees put the blocks together to make a
 solid base. 
When they finished their treehouse, Goodger and the trees sat inside. Badger said, "Look how fancy we made it!"
The trees smiled and nodded. They said, "It's very fancy! Thank you for helping us organize this operation." 
Then they lived happily in their fancy treehouse together!
<s>
Once upon a time, there was a boy named Timmy. Timmy was very hungry and wanted to eat his meal. He asked his mom, "What are we having for dinner
?" His mom said, "We are having chicken and rice." Timmy said, "Yum! I love chicken and rice."
While they were eating, Timmy's dad came in and said, "Hey Timmy, do you want to watch a movie after
2023/07/29 09:30:58 achieved tok/s: 28.619646
```

### Performance

| system                  | model           | llama2.c      | llama.cpp          | llama2.go[^simple] | llama2.go[^fast] |
| ------------------------| --------------- | ------------: | -----------------: | -----------------: | ---------------: |
| Apple M1 Max 10CPU 64GB | stories110M     |  101.84 tok/s |                    |        10.47 tok/s |      39.28 tok/s |  
| Apple M1 Max 10CPU 64GB | llama2_7b       |    1.83 tok/s |        20.36 tok/s |                    |       0.87 tok/s | 
| Apple M1 Max 10CPU 64GB | llama2_13b      |    (segfault) |        11.71 tok/s |                    |       0.38 tok/s |

### Optimizations

* transformer steps parallelism
* loop unrolling
* in-matrix parallelism
* (todo) SIMD
* (todo) quantization

All optimizations are `Fuzz`-tested against basic algorithm, which is itself tested.
To disable optimizations update `llama2/transformer.go` import to package without optimizations and rebuild.

### Related Work and References

* https://github.com/karpathy/llama2.c
* https://github.com/poudels14/llama2_rs (`llama2.c` Rust port)
* https://github.com/saracen/llama2.go (`llama2.c` Go port, `mmap`, `cgo`)
* https://github.com/tmc/go-llama2 (`llama2.c` Go fork)
* https://github.com/haormj/llama2.go (`llama2.c` Go port, `cobra`)
* https://github.com/gotzmann/llama.go (`llama.cpp` port in Go)
* https://github.com/go-skynet/go-llama.cpp (`cgo` bidning `llama.cpp`)
* https://github.com/go-skynet/LocalAI (`cgo` binding API of many models)
* https://github.com/ggerganov/llama.cpp
* [RoPE](https://arxiv.org/pdf/2104.09864.pdf)
* ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)


[^simple]: No linear algebra optimizations
[^fast]: All linear algebra optimizations
