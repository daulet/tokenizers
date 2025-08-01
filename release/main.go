package main

import (
	"fmt"

	"github.com/daulet/tokenizers/release/tokenizers"
)

func chatTemplateDeepSeek() error {
	template := "test/data/deepseek-ai/DeepSeek-R1/tokenizer_config.json"
	ct, err := tokenizers.NewChatTemplate(template)
	if err != nil {
		fmt.Printf("NewChatTemplate error: %v\n", err)
		return err
	}
	defer ct.Close()

	messages_str := `[{"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "How are you?"
        },
		{
			"role": "assistant",
			"content": "Hello! How can I assist you today?"
		},
		{
			"role": "user",
			"content": "What can you do?"
		}
    ]`

	result, err := ct.ApplyChatTemplate(messages_str, "", "")
	if err != nil {
		fmt.Printf("Failed to apply chat template: %v", err)
		return err
	}
	fmt.Printf("apply chat_template for DeepSeek: %v", result)
	return nil
}

func chatTemplateQwen3() error {

	template := "test/data/Qwen/Qwen3-235B-A22B/tokenizer_config.json"
	ct, err := tokenizers.NewChatTemplate(template)
	if err != nil {
		fmt.Printf("Failed to create chat template: %v", err)
		return err
	}
	defer ct.Close()

	messages_str := `[{"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "How are you?"
        },
		{
			"role": "assistant",
			"content": "Hello! How can I assist you today?"
		},
		{
			"role": "user",
			"content": "What can you do?"
		}
    ]`

	result, err := ct.ApplyChatTemplate(messages_str, "", "")
	if err != nil {
		fmt.Printf("Failed to apply chat template: %v", err)
	}
	fmt.Printf("apply chat_template for qwen3: %v", result)
	return nil
}

func main() {
	tk, err := tokenizers.FromFile("./test/data/bert-base-uncased.json")
	if err != nil {
		panic(err)
	}
	// release native resources
	defer tk.Close()
	fmt.Println("Vocab size:", tk.VocabSize())
	// Vocab size: 30522
	// [2829 4419 14523 2058 1996 13971 3899] [brown fox jumps over the lazy dog]
	fmt.Println(tk.Encode("brown fox jumps over the lazy dog", true))
	// [101 2829 4419 14523 2058 1996 13971 3899 102] [[CLS] brown fox jumps over the lazy dog [SEP]]
	fmt.Println(tk.Decode([]uint32{111308, 3837, 35946, 101922, 30534, 100134, 104811}, true))
	// brown fox jumps over the lazy dog

	if err = chatTemplateDeepSeek(); err != nil {
		panic(err)
	}
	if err = chatTemplateQwen3(); err != nil {
		panic(err)
	}

}
