from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


def build_regex_tokenizer(
    feature: list,
    regex_string: str,
    tokenizer_behaviour: str = "isolated",
    max_vocab_size: int = 10000,
    max_length: int = 512,
):
    mol_pre_tok = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                pattern=Regex(regex_string),
                behavior=tokenizer_behaviour,
            )
        ]
    )

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = mol_pre_tok

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    trainer = trainers.WordLevelTrainer(
        vocab_size=max_vocab_size, special_tokens=special_tokens
    )

    tok.train_from_iterator(feature, trainer=trainer)

    eos_token_id = tok.token_to_id("<eos>")
    bos_token_id = tok.token_to_id("<bos>")
    tok.post_processor = processors.TemplateProcessing(
        single=f"<bos>:0 $A:0 <eos>:0",
        special_tokens=[
            ("<bos>", bos_token_id),
            ("<eos>", eos_token_id),
        ],
    )

    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        model_max_length=max_length,
    )

    return wrapped_tokenizer
