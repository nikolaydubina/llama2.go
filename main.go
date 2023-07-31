package main

import (
	"flag"
	"log"
	"os"
	"time"

	"github.com/nikolaydubina/llama2.go/llama2"
	"github.com/nikolaydubina/llama2.go/nn"
)

func main() {
	var (
		checkpointFilePath string
		tokenizerFilePath  string
		temperature        float64
		steps              int
		prompt             string
	)

	flag.StringVar(&checkpointFilePath, "checkpoint", "out/model.bin", "checkpoint binary file with weights")
	flag.StringVar(&tokenizerFilePath, "tokenizer", "tokenizer.bin", "tokenizer binary file with vocabulary (get this from repo)")
	flag.Float64Var(&temperature, "temperature", 0.9, "temperature is optional, 0 = (deterministic) argmax sampling, 1 = baseline")
	flag.IntVar(&steps, "steps", 256, "max number of steps to run for, 0: use seq_len")
	flag.StringVar(&prompt, "prompt", "", "query to start with")
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

	promptTokens := vocab.Encode(prompt)

	// the current position we are in
	timeStart := time.Now()
	var token int = 1          // 1 = BOS token in llama-2 sentencepiece
	out.Write([]byte("<s>\n")) // explicit print initial BOS token for stylistic symmetry reasons
	for pos := 0; pos < steps; pos++ {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, config, runState, w)

		var next int
		if pos < len(promptTokens) {
			next = promptTokens[pos]
		} else {
			// sample the next token
			if temperature == 0 {
				// greedy argmax sampling
				next = nn.ArgMax(runState.Logits)
			} else {
				// apply the temperature to the logits
				for q := 0; q < config.VocabSize; q++ {
					runState.Logits[q] /= float32(temperature)
				}
				// apply softmax to the logits to the probabilities for next token
				nn.SoftMax(runState.Logits)
				// we now want to sample from this distribution to get the next token
				next = nn.Sample(runState.Logits)
			}
		}

		// following BOS token (1), sentencepiece decoder strips any leading whitespace
		var tokenStr string
		if token == 1 && vocab.Words[next][0] == ' ' {
			tokenStr = vocab.Words[next][1:]
		} else {
			tokenStr = vocab.Words[next]
		}
		out.WriteString(tokenStr)

		// advance forward
		token = next
	}
	out.Write([]byte("\n"))

	log.Printf("achieved tok/s: %f\n", float64(steps)/time.Since(timeStart).Seconds())
}
