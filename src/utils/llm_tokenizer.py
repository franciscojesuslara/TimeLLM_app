from transformers import GPT2Config, GPT2Model, GPT2Tokenizer, LlamaTokenizer, LlamaModel, LlamaConfig, AutoTokenizer, \
    BertModel, BertTokenizer,BertConfig

def select_llm(name_llm: str):
    if name_llm == 'gpt':
        llm_config = GPT2Config.from_pretrained('openai-community/gpt2')
        llm_model = GPT2Model.from_pretrained('openai-community/gpt2', config=llm_config)
        llm_tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')

    elif name_llm == 'llama3':
        llm_config = LlamaConfig.from_pretrained(
                            'meta-llama/Meta-Llama-3-8B',
                    token='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',
                    trust_remote_code=True,
                    local_files_only=False)
        llm_model = LlamaModel.from_pretrained(
                            'meta-llama/Meta-Llama-3-8B',
                    token='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',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=llm_config)
        llm_tokenizer = AutoTokenizer.from_pretrained(
                            'meta-llama/Meta-Llama-3-8B',
                    token='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',
                    trust_remote_code=True,
                    local_files_only=False)
    elif name_llm == 'llama2':
        llm_config = LlamaConfig.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token='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',
            trust_remote_code=True,
            local_files_only=False)
        llm_model = LlamaModel.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token='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',
            trust_remote_code=True,
            local_files_only=False,
            config=llm_config)
        llm_tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-2-7b-chat-hf',
            token='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',
            trust_remote_code=True,
            local_files_only=False)

    elif name_llm == 'bert':
        llm_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

        llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=llm_config,
            )
        llm_tokenizer= BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False)
    return llm_config, llm_model, llm_tokenizer