import sys

from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM


def main():
    """Run a simple T5 text-generation example."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
    except Exception as exc:
        print("Failed to load T5 model:", exc, file=sys.stderr)
        sys.exit(1)

    prompt = "나는 한국에서 산다."

    try:
        input_ids = tokenizer(prompt, return_tensors="tf").input_ids
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
        )
        output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    except Exception as exc:
        print("Generation error:", exc, file=sys.stderr)
        sys.exit(1)

    print(output)


if __name__ == "__main__":
    main()
