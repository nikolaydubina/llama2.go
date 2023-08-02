package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
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

	tokenizer := llama2.NewTokenizerFromFile(config.VocabSize, tokenizerFile)

	w := llama2.NewTransformerWeightsFromCheckpoint(config, checkpointFile, isSharedWeights)

	// right now we cannot run for more than config.SeqLen steps
	if steps <= 0 || steps > config.SeqLen {
		steps = config.SeqLen
	}

	runState := llama2.NewRunState(config)

	promptTokens := tokenizer.Encode(prompt)

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
		if token == 1 && tokenizer.Words[next][0] == ' ' {
			tokenStr = tokenizer.Words[next][1:]
		} else {
			tokenStr = tokenizer.Words[next]
		}
		out.WriteString(tokenStr)

		// advance forward
		token = next
	}
	out.Write([]byte("\n"))

	log.Printf("achieved tok/s: %f\n", float64(steps)/time.Since(timeStart).Seconds())
}

const (
	B_INST = "[INST]"
	E_INST = "[/INST]"
	B_SYS  = "<<SYS>>\n"
	E_SYS  = "\n<</SYS>>\n\n"
)

// go:embed dialog_default_system_prompt.txt
var DefaultSystemPrompt string

type Role uint

const (
	System Role = iota + 1
	User
	Assistant
)

func (s Role) String() string {
	switch s {
	case System:
		return "system"
	case User:
		return "user"
	case Assistant:
		return "assistant"
	default:
		return ""
	}
}

type Message struct {
	Role    Role
	Content string
}

type Dialog []Message

func RunDialog(
	config llama2.Config,
	dialogs []Dialog,
	maxGenLen int,
) (response Dialog, error) {
	if maxGenLen == 0 {
		maxGenLen = config.SeqLen - 1
	}

	var promptTokens []int

	// creating basic dialog
	for _, dialog := range dialogs {
		if dialog[0].Role != System {
			dialog = append(Dialog{{Role: System, Content: DefaultSystemPrompt}}, dialog...)
		}
		dialog = append(Dialog{{Role: dialog[1].Role, Content: B_SYS + dialog[0].Content + E_SYS + dialog[1].Content}}, dialog[2:]...)

		// dialog roles check
		for i, m := range dialog {
			var expRole Role = System
			if i%2 == 0 {
				expRole = User
			}
			if m.Role != expRole {
				return fmt.Errorf("expected roles User/System/User/..., at i(%d) expected(%s) but got role(%s)", i, expRole, m.Role)
			}
		}

		if lastRole := dialog[len(dialog)-1].Role; lastRole != User {
			return fmt.Errorf("last prompt should be from role(%s) but got role(%s)", User, lastRole)
		}

		// collect dialog tokens
		var dialogTokens []int
		for i := 0; i < len(dialog); i += 2 {
			prompt := dialog[i]
			answer := dialog[i+1]

			var message strings.Builder
			message.WriteString(B_INST)
			message.WriteRune(' ')
			message.WriteString(strings.TrimSpace(prompt.Content))
			message.WriteRune(' ')
			message.WriteString(E_INST)
			message.WriteString(strings.TrimSpace(answer.Content))

			dialogTokens = append(dialogTokens, tokenizer.Encode2(message.String(), true /* bos */, true /* eos */)...)
		}

		var message strings.Builder
		message.WriteString(B_INST)
		message.WriteRune(' ')
		message.WriteString(strings.TrimSpace(dialog[len(dialog)-1].Content))
		message.WriteRune(' ')
		message.WriteString(E_INST)
		dialogTokens = append(dialogTokens, tokenizer.Encode2(message.String(), true /* bos */, false /* eos */)...)

		promptTokens = append(promptTokens, dialogTokens...)
	}

	generationTokens := generate()

	for _, t := range generationTokens {
		response = append(response, Message{Role: Assistant, Content: tokenizer.Decode(t)})
	} 

	return response, nil
}
