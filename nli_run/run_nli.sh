# download MNLI:'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce'
# download QNLI:'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0'

# Download models from https://huggingface.co/models and put them in local path. 
# E.g., download roberta-base from https://huggingface.co/roberta-base
# E.g., download roberta-large from https://huggingface.co/roberta-large

# The directory where the datasets and models are located.
cache_dir=$1

mnli_data_dir=cache_dir+"MNLI"
qnli_data_dir=cache_dir+"QNLI"

roberta_base_dir=cache_dir+"roberta-base"
roberta_large_dir=cache_dir+"roberta-large"

roberta_base_mnli_dir=cache_dir+"roberta-base-mnli"
roberta_base_qnli_dir=cache_dir+"roberta-base-qnli"

roberta_large_mnli_dir=cache_dir+"roberta-large-mnli"
roberta_large_qnli_dir=cache_dir+"roberta-large-qnli"

# roberta-base-mnli
python run_nli_roberta.py --data_dir=${mnli_data_dir} --roberta_model_dir=${roberta_base_dir} --task_name="mnli" --output_dir=${roberta_base_mnli_dir} --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3 --warmup_proportion=0.15 --gpuid=0

# roberta-base-qnli
python run_nli_roberta.py --data_dir=${qnli_data_dir} --roberta_model_dir=${roberta_base_dir} --task_name="qnli" --output_dir=${roberta_base_qnli_dir} --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3 --warmup_proportion=0.15 --gpuid=0

# roberta-large-mnli
python run_nli_roberta.py --data_dir=${mnli_data_dir} --roberta_model_dir=${roberta_large_dir} --task_name="mnli" --output_dir=${roberta_large_mnli_dir} --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --learning_rate=3e-5 --num_train_epochs=3 --warmup_proportion=0.15 --gpuid=0

# roberta-large-qnli
python run_nli_roberta.py --data_dir=${qnli_data_dir} --roberta_model_dir=${roberta_large_dir} --task_name="qnli" --output_dir=${roberta_large_qnli_dir} --max_seq_length=128 --do_train --do_eval --train_batch_size=32 --learning_rate=3e-5 --num_train_epochs=3 --warmup_proportion=0.15 --gpuid=0