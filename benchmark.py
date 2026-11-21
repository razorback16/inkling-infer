import argparse
import statistics
import mlx.core as mx
from mlx_lm import load, stream_generate
from inkling_kernel import install

parser = argparse.ArgumentParser()
parser.add_argument("--kernel", action="store_true")
args = parser.parse_args()
if args.kernel:
    install()
model, tokenizer = load("thinkingmachines/Inkling-2-Mini")
prompt = tokenizer.encode("def fibonacci(n):\n    ") * 16
rates = []
for _ in range(4):
    for result in stream_generate(model, tokenizer, prompt, max_tokens=96):
        pass
    rates.append(result.generation_tps)
print(statistics.median(rates[1:]))
