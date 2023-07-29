## llama2.go

[![Go Report Card](https://goreportcard.com/badge/github.com/nikolaydubina/llama2.go)](https://goreportcard.com/report/github.com/nikolaydubina/llama2.go)
[![codecov](https://codecov.io/gh/nikolaydubina/llama2.go/branch/master/graph/badge.svg?token=OMf0git2BD)](https://codecov.io/gh/nikolaydubina/llama2.go)
[![Go Reference](https://pkg.go.dev/badge/github.com/nikolaydubina/llama2.go.svg)](https://pkg.go.dev/github.com/nikolaydubina/llama2.go)

Native Go version of [llama2.c](https://github.com/karpathy/llama2.c).

It is pure Go inference code ported from experimental implementation by [Andrej Karpathy](https://en.wikipedia.org/wiki/Andrej_Karpathy) of latest as of `2023-07-25` LLM model from Meta [LLAMA-2](https://ai.meta.com/llama/).  

### How to run?

1. get `tokenizer.bin` from [llama2.c](https://github.com/karpathy/llama2.c) (included)
2. get weights `wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin`
3. `go install github.com/nikolaydubina/llama2.go@latest`
4. `llama2.go -checkpoint=stories110M.bin -prompt="good morning said sun to trees"`

```
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
````

### Differences from `llama2.c`

* for checkpoint not using `mmap`, instead scanning file

### Performance

| model           | llama2.c          | llama2.go
| --------------- | ----------------- | ----------------
| stories42M.bin  |  265.348595 tok/s | 25.677383  tok/s
| stories110M.bin |  101.837061 tok/s | 10.474615  tok/s

### Optimizations

* transformer steps parallelism
* (experimental) loop unrolling
* (experimental) in-matrix parallelism
* (todo) SIMD

| model           | llama2.c          | llama2.go
| --------------- | ----------------- | ----------------
| stories42M.bin  |  265.348595 tok/s | 82.793488  tok/s
| stories110M.bin |  101.837061 tok/s | 39.280158  tok/s

To enable experimental optimizations update `llama2/transformer.go` import to use package with optimizations and rebuild.

```go
package llama2

import (
	"math"
	"sync"

	nn "github.com/nikolaydubina/llama2.go/exp/nnfast"
)
```

### Related Work

* https://github.com/karpathy/llama2.c
* https://github.com/poudels14/llama2_rs
* https://github.com/saracen/llama2.go (`mmap`) — another very good llama2.go port
* https://github.com/tmc/go-llama2 (fork)
* https://github.com/haormj/llama2.go (cobra)
* https://github.com/gotzmann/llama.go
