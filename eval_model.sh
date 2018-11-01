cd <path-to-ParlAI-repo>
python parlai/scripts/eval_model.py \
--task convai2:self \
--model-file model_file_path # PUT YOUR MODEL HERE
-dt valid \
--beam-size 5 \
--batchsize 32 \

# this eval is doing simple beam search with minibatch size 32.
