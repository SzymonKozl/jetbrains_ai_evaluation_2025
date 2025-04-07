import transformers
from typing import List, Tuple
import evaluate
from sys import argv
from json import load
from codebleu import calc_codebleu


TAG_PREFIX = '<fim_prefix>'
TAG_SUFFIX = '<fim_suffix>'
TAG_MIDDLE = '<fim_middle>'


def complete_sentence(model: transformers.AutoModelForCausalLM, tokenizer: transformers.AutoTokenizer, prefix: str, suffix: str):
    input_str = f"{TAG_PREFIX}{prefix} {TAG_SUFFIX}{suffix}{TAG_MIDDLE}"
    inputs = tokenizer.encode(input_str, return_tensors='pt')
    return tokenizer.decode(model.generate(inputs)[0]).split(TAG_MIDDLE)[1]


def score_on_examples(model, tokenizer, examples: List[dict], first_line_only: bool) -> Tuple[dict, List[str]]:
    metric_chrf =evaluate.load('chrf')
    predictions = []
    references = []
    for example in examples:
        prefix = example['prefix']
        suffix = example['suffix']
        masked = example['missing']
        masked_pred = complete_sentence(model, tokenizer, prefix, suffix)
        if first_line_only:
            masked_pred = masked_pred.split('\n')[0] + '\n'
        predictions.append(masked_pred)
        references.append([masked])
    bleu = calc_codebleu(predictions, [x[0] for x in references], lang='python')
    chrf = metric_chrf.compute(predictions=predictions, references=references)
    return {'bleu': bleu, 'chrf': chrf['score']}, predictions


def eval_model(model_checkpoint, examples: List[dict], first_line_only: bool = False) -> Tuple[dict, List[str]]:
    """
    Evaluates the model loaded from checkpoint on the given examples using chrf metric.
    first_line_only specifies whether to cut model's  output at the first newline character.
    Returns the score and a list of predictions.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_checkpoint)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_checkpoint)
    metrics, predictions = score_on_examples(model, tokenizer, examples, first_line_only)
    return metrics, predictions


if __name__ == '__main__':
    assert len(argv) == 3, f"usage: {argv[0]} <model_name> <examples_path>"
    model_name = argv[1]
    examples_path = argv[2]
    with open(examples_path, 'r') as f:
        examples = load(f)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    print(eval_model(model_name, examples)[0])