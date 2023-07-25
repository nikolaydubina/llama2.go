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
$ go run main.go -temperature=0 -checkpoint out44m/model44m.bin
2023/07/26 01:52:56 config: llama2.Config{Dim:512, HiddenDim:1376, NumLayers:8, NumHeads:8, NumKVHeads:8, VocabSize:32000, SeqLen:1024}
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went for a walk in the forest with her mommy. They saw many trees and flowers. Suddenly, Lily saw a big, scary bear! She was very scared and didn't know what to do.
Her mommy told her to stay still and not move. The bear sniffed around them and then walked away. Lily was relieved and happy that she didn't get hurt. Her mommy told her that it's important to always stay safe and not wander off alone.
From that day on, Lily always remembered to stay close to her mommy when they went for walks in the forest. She learned that it's important to be careful and not wander off alone.
<s>
 Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she went for a walk in the forest with her mommy. They saw many trees and flowers. Suddenly, Lily saw a big, scary bear! She was very scared and didn't know what to
2023/07/26 01:53:09 achieved tok/s: 80.839978
```

## Differences from `llama2.c`

* for checkpoint not using `mmap`, instead scanning file

## Performance

```
llama.c
achieved tok/s: 125.860374

llama.go
achieved tok/s: 80.839978
```

## References

* https://github.com/karpathy/llama2.c
* https://github.com/poudels14/llama2_rs
