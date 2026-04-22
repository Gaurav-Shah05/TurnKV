modal secret create hf-secret --force HF_TOKEN=$HF_TOKEN

MODAL_HF_SECRET_NAME=hf-secret modal run -d evaluation/benchmarks/convcodeworld/modal_app.py::main \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--press-names snapkv \
--compression-ratio 0.5 \
--fraction 0.1 \
--num-eval-examples -1 \
--local-budget 4096

Floor-only variant:

MODAL_HF_SECRET_NAME=hf-secret modal run -d evaluation/benchmarks/convcodeworld/modal_app.py::main \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--press-names snapkv \
--fraction 0.1 \
--num-eval-examples -1 \
--local-budget 4096 \
--alpha-floor 1.0