## llama2.go

Native Go version of [llama2.c](https://github.com/karpathy/llama2.c).

It is pure Go inference code ported from experimental implementation by [Andrej Karpathy](https://en.wikipedia.org/wiki/Andrej_Karpathy) of latest as of `2023-07-25` LLM model from Meta [LLAMA-2](https://ai.meta.com/llama/).  

### How to run?

1. get `tokenizer.bin` from [llama2.c](https://github.com/karpathy/llama2.c) (included)
2. get weights from from [llama2.c](https://github.com/karpathy/llama2.c). `wget https://karpathy.ai/llama2c/model44m.bin -P out44m`
3. `go install github.com/nikolaydubina/llama2.go@latest`
4. `llama2.go -checkpoint out44m/model44m.bin`

Example output:

```
$ llama2.go -temperature=0.1 -checkpoint out44m/model44m.bin
2023/07/26 02:50:05 config: llama2.Config{Dim:512, HiddenDim:1376, NumLayers:8, NumHeads:8, NumKVHeads:8, VocabSize:32000, SeqLen:1024}
 One day, a little girl named Amy went to the park. She saw a big tree with a lot of fruit. The fruit was red, yellow, and green. Amy wanted to eat the fruit, but she was too small to reach it. She felt sad.
A tall boy named Tom saw Amy and asked, "Why are you sad?" Amy said, "I want to eat the fruit, but I am too small to reach it." Tom was a creative boy. He thought for a moment and said, "Let's think of a way to get the fruit."
They tried to jump and climb, but they still could not reach the fruit. Then, Tom had an idea. He said, "Let's ask the tree
 for help." They asked the tree, and the tree shook its leaves. The fruit fell down, and Amy could reach it.
Amy and Tom were very happy. They shared the fruit and became good friends. The tree was happy too, because it could help them.
<s>
 Once upon a time, there was a big house with a big lawn. The lawn was so large that it looked like an enormous ball. One day, a little boy came to play on the lawn.
2023/07/26 02:50:18 achieved tok/s: 79.993750
````

### Differences from `llama2.c`

* for checkpoint not using `mmap`, instead scanning file

### Performance

```
llama.c
achieved tok/s: 125.860374

llama.go
achieved tok/s: 80.839978
```

### Issues

* temperature is too sensitive. to get good results set temperature < 0.2. Original `llama2.c` handles temperatures much better. Using `float64` in Go does not help. Investigation why this is so needed.

### Related Work

* https://github.com/karpathy/llama2.c
* https://github.com/poudels14/llama2_rs
* https://github.com/gotzmann/llama.go
* https://github.com/tmc/go-llama2
* https://github.com/haormj/llama2.go
