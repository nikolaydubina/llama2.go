package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/nikolaydubina/llama2.go/llama2"
)

func main() {
	var (
		checkpointFilePath string
		tokenizerFilePath  string
		temperature        float64
		steps              int
	)

	flag.StringVar(&checkpointFilePath, "checkpoint", "out/model.bin", "checkpoint binary file with weights")
	flag.StringVar(&tokenizerFilePath, "tokenizer", "tokenizer.bin", "tokenizer binary file with vocabulary (get this from llama2.c repo)")
	flag.Float64Var(&temperature, "temperature", 0.9, "temperature is optional, 0 = (deterministic) argmax sampling, 1 = baseline")
	flag.IntVar(&steps, "steps", 256, "max number of steps to run for, 0: use seq_len")
	flag.Parse()

	r, err := os.OpenFile(checkpointFilePath, os.O_RDONLY, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	out := os.Stdout

	config := llama2.NewConfigFromCheckpoint(r)

	// "negative vocab size is hacky way of signaling unsahred weights. biy yikes" — @karpathy
	isSharedWeights := config.VocabSize > 0
	if config.VocabSize < 0 {
		config.VocabSize = -config.VocabSize
	}

	tokenizerFile, err := os.OpenFile(tokenizerFilePath, os.O_RDONLY, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer tokenizerFile.Close()

	vocab := llama2.NewVocabFromFile(config.VocabSize, r)

	w := llama2.NewTransformerWeightsFromCheckpoint(config, r, isSharedWeights)

	// right now we cannot run for more than config.SeqLen steps
	if steps <= 0 || steps > config.SeqLen {
		steps = config.SeqLen
	}

	runState := llama2.NewRunState(config)

	// the current position we are in
	timeStart := time.Now()
	var next int
	var token int = 1 // 1 = BOS token in llama-2 sentencepiece
	var pos int = 0
	for pos < steps {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, config, runState, w)

		// sample the next token
		if temperature == 0 {
			// greedy maxarg sampling
			next = llama2.ArgMax(runState.Logits)
		} else {
			// apply the temperature to the logits
			for q := 0; q < config.VocabSize; q++ {
				runState.Logits[q] /= float32(temperature)
			}
			// apply softmax to the logits to the probabilities for next token
			llama2.SoftMax(runState.Logits)
			// we now want to sample from this distribution to get the next token
			next = llama2.Sample(runState.Logits)
		}
		out.WriteString(vocab[next])

		// advance forward
		token = next
		pos++
	}

	log.Printf("achieved tok/s: %f\n", 1000*float64(config.SeqLen)/float64(time.Since(timeStart).Milliseconds()))
}
