export HF_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-8B
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-8B-Instruct
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-70B
# huggingface-cli download --token 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 --resume-download meta-llama/Meta-Llama-3-70B-Instruct --local-dir Llama-3-70B-Instruct --local-dir-use-symlinks False
# huggingface-cli download --resume-download Qwen/Qwen2-7B-Instruct
huggingface-cli download --resume-download 01-ai/Yi-1.5-9B-Chat
huggingface-cli download --resume-download 01-ai/Yi-1.5-34B-Chat
# huggingface-cli download --resume-download 01-ai/Yi-1.5-9B-Chat-16K
# huggingface-cli download --resume-download 01-ai/Yi-1.5-34B-Chat-16K
# huggingface-cli download --repo-type dataset --resume-download MMInstruction/VLFeedback --local-dir data/VLFeedback --local-dir-use-symlinks False