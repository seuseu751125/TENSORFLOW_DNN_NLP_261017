import keras_hub


def main():
    """Generate text with a T5 model from KerasHub."""
    prompt = "나는 한국에서 산다."
    tokenizer = keras_hub.models.T5Tokenizer.from_preset("t5_base_en")
    model = keras_hub.models.T5Backbone.from_preset("t5_base_en")

    input_ids = tokenizer(prompt)
    generated_ids = model.generate(
        input_ids,
        max_length=50,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=2,
    )
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(output)


if __name__ == "__main__":
    main()
