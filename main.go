package main

import (
	"bufio"
	_ "embed"
	"flag"
	"fmt"
	"io"
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
		isDialog           bool
	)

	flag.StringVar(&checkpointFilePath, "checkpoint", "out/model.bin", "checkpoint binary file with weights")
	flag.StringVar(&tokenizerFilePath, "tokenizer", "tokenizer.bin", "tokenizer binary file with vocabulary (get this from repo)")
	flag.Float64Var(&temperature, "temperature", 0.9, "temperature is optional, 0 = (deterministic) argmax sampling, 1 = baseline")
	flag.IntVar(&steps, "steps", 256, "max number of steps to run for, 0: use seq_len")
	flag.StringVar(&prompt, "prompt", "", "query to start with")
	flag.BoolVar(&isDialog, "dialog", false, "run interactive dialog mode")
	flag.Parse()

	if isDialog && prompt != "" {
		log.Fatal("cannot use -dialog and -prompt together")
	}

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

	defer func(start time.Time) { log.Printf("achieved tok/s: %f\n", float64(steps)/time.Since(start).Seconds()) }(time.Now())

	if !isDialog {
		var promptTokens []int
		promptTokens = append(promptTokens, tokenizer.BOS_ID)
		promptTokens = append(promptTokens, tokenizer.Encode(prompt)...)
		generate(config, runState, w, tokenizer.Decoder(out), temperature, promptTokens, steps)
	} else {
		RunDialog(config, tokenizer, runState, w, nil, out, steps, temperature)
	}
}

// generate forwards model through prompt and generates up to maxGenLen tokens after it
func generate(
	config llama2.Config,
	runState llama2.RunState,
	w llama2.TransformerWeights,
	tokenizerDecoder *llama2.TokenDecoder,
	temperature float64,
	promptTokens []int,
	maxGenLen int,
) {
	for token, pos := promptTokens[0], 0; pos < maxGenLen; pos++ {
		// forward the transformer to get logits for the next token
		llama2.Transformer(token, pos, config, runState, w)

		// different strategies to pick next token from probabilities or current prompt
		if pos < len(promptTokens) {
			token = promptTokens[pos]
		} else {
			// sample the next token
			if temperature == 0 {
				// greedy argmax sampling
				token = nn.ArgMax(runState.Logits)
			} else {
				// apply the temperature to the logits
				for q := 0; q < config.VocabSize; q++ {
					runState.Logits[q] /= float32(temperature)
				}
				// apply softmax to the logits to the probabilities for next token
				nn.SoftMax(runState.Logits)
				// we now want to sample from this distribution to get the next token
				token = nn.Sample(runState.Logits)
			}
		}

		tokenizerDecoder.WriteToken(token)
	}
}

const (
	B_INST = "[INST]"
	E_INST = "[/INST]"
	B_SYS  = "<<SYS>>"
	E_SYS  = "<</SYS>>"
)

//go:embed dialog_system_prompt_default.txt
var DialogSystemPromptDefault string

//go:embed dialog_system_prompt_short.txt
var DialogSystemPromptShort string

// Role tells whose message it is in dialog. LLaMA2 has also Assistant, that is not used here.
type Role string

const (
	System Role = "system"
	User   Role = "user"
)

type Message struct {
	Role    Role
	Content string
}

func (m Message) String() string { return fmt.Sprintf("%s: %s", m.Role, m.Content) }

// RunDialog transforms dialog messages into prompts for model to predict, boots dialog with standard messages.
func RunDialog(
	config llama2.Config,
	tokenizer llama2.Tokenizer,
	runState llama2.RunState,
	w llama2.TransformerWeights,
	dialog []Message,
	out io.StringWriter,
	maxGenLen int,
	temperature float64,
) {
	if maxGenLen == 0 {
		maxGenLen = config.SeqLen - 1
	}

	// dialog startup
	if len(dialog) == 0 || dialog[0].Role != System {
		dialog = append([]Message{{Role: System, Content: DialogSystemPromptShort}}, dialog...)
	}
	dialog = append([]Message{{Role: dialog[1].Role, Content: B_SYS + dialog[0].Content + E_SYS + dialog[1].Content}}, dialog[2:]...)

	// dialog roles check
	for i, m := range dialog {
		var expRole Role = System
		if i%2 == 0 {
			expRole = User
		}
		if m.Role != expRole {
			panic(fmt.Errorf("expected roles User/System/User/..., at i(%d) expected(%s) but got role(%s)", i, expRole, m.Role))
		}
	}

	if lastRole := dialog[len(dialog)-1].Role; lastRole != User {
		panic(fmt.Errorf("last prompt should be from role(%s) but got role(%s)", User, lastRole))
	}

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		prompt := scanner.Text()
		dialog := []Message{{Role: User, Content: prompt}}
	}

	// collect dialog tokens
	var dialogTokens []int
	for i := 0; i+1 < len(dialog); i += 2 {
		prompt := dialog[i]
		answer := dialog[i+1]

		dialogTokens = append(dialogTokens, tokenizer.BOS_ID)
		dialogTokens = append(dialogTokens, tokenizer.Encode(B_INST+" "+prompt.String()+" "+E_INST+strings.TrimSpace(answer.Content))...)
		dialogTokens = append(dialogTokens, tokenizer.EOS_ID)
	}

	dialogTokens = append(dialogTokens, tokenizer.BOS_ID)
	dialogTokens = append(dialogTokens, tokenizer.Encode(B_INST+" "+strings.TrimSpace(dialog[len(dialog)-1].Content)+" "+E_INST)...)

	generate(config, runState, w, tokenizer.Decoder(out), temperature, dialogTokens, maxGenLen)
}
