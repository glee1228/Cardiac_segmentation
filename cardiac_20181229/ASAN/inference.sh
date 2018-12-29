mkdir /data/output/CHD
mkdir /data/output/HCMP

python3 src/inference.py --data_dir='/data/test/CHD' --output_dir='/data/output/CHD' --image_filename='_dia.mha' --model_path='/data/model/CHD_dia/checkpoint.meta' --checkpoint_path='/data/model/CHD_dia/checkpoint'
python3 src/inference.py --data_dir='/data/test/CHD' --output_dir='/data/output/CHD' --image_filename='_dia.mha' --model_path='/data/model/CHD_sys/checkpoint.meta' --checkpoint_path='/data/model/CHD_sys/checkpoint'
python3 src/inference.py --data_dir='/data/test/HCMP' --output_dir='/data/output/HCMP' --image_filename='A_' --model_path='/data/model/HCMP_A/checkpoint.meta' --checkpoint_path='/data/model/HCMP_A/checkpoint'
python3 src/inference.py --data_dir='/data/test/HCMP' --output_dir='/data/output/HCMP' --image_filename='N_' --model_path='/data/model/HCMP_N/checkpoint.meta' --checkpoint_path='/data/model/HCMP_N/checkpoint'

