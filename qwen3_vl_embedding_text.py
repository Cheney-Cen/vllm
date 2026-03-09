from vllm import LLM

def main():
    model_dir = "/Users/chenchen/Downloads/Qwen3-VL-Embedding-0.6B-20260216-32x-step110000/checkpoint_step_110000"

    llm = LLM(
        model=model_dir,
        runner="pooling",
        max_model_len=8192,
        trust_remote_code=True,
        disable_log_stats=True,
    )

    (output,) = llm.embed("你好，世界")
    emb = output.outputs.embedding
    print("len:", len(emb))
    print("first 8 dims:", emb[:8])

if __name__ == "__main__":
    main()