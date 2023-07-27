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
	flag.Float64Var(&temperature, "temperature", 0.1, "temperature is optional, 0 = (deterministic) argmax sampling, 1 = baseline. as of 2023-07-26, use low temperature for good results. original llama2.c handles temperature much better.")
	flag.IntVar(&steps, "steps", 256, "max number of steps to run for, 0: use seq_len")
	flag.Parse()

	checkpointFile, err := os.OpenFile(checkpointFilePath, os.O_RDONLY, 0)
	if err != nil {
		log.Fatal(err)
	}
	defer checkpointFile.Close()

	out := os.Stdout

	config, err := llama2.NewConfigFromCheckpoint(checkpointFile)
	if err != nil {
		log.Fatalf("cannot read config: %s", err)
	}
	log.Printf("config: %#v\n", config)

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

	vocab := llama2.NewVocabFromFile(config.VocabSize, tokenizerFile)

	w := llama2.NewTransformerWeightsFromCheckpoint(config, checkpointFile, isSharedWeights)

	// right now we cannot run for more than config.SeqLen steps
	if steps <= 0 || steps > config.SeqLen {
		steps = config.SeqLen
	}

	runState := llama2.NewRunState(config)

	// the current position we are in
	timeStart := time.Now()
	var next int
	var token int = 1 // 1 = BOS token in llama-2 sentencepiece
	for pos := 0; pos < steps; pos++ {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, config, runState, w)

		// sample the next token
		if temperature == 0 {
			// greedy argmax sampling
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
	}
	out.Write([]byte("\n"))

	log.Printf("achieved tok/s: %f\n", float64(config.SeqLen)/time.Since(timeStart).Seconds())
}
