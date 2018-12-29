mkdir /data/output/CHD
mkdir /data/output/HCMP

python3 src/inference.py --test_path='/data/test/CHD' --output_path='/data/output/CHD' --test_filename='_dia.mha' --model_name='/data/model/CHD_dia/checkpoint_chd_dia.h5'
python3 src/inference.py --test_path='/data/test/CHD' --output_path='/data/output/CHD' --test_filename='_sys.mha' --model_name='/data/model/CHD_sys/checkpoint_chd_sys.h5' 
python3 src/inference.py --test_path='/data/test/HCMP' --output_path='/data/output/HCMP' --test_filename='A_' --model_name='/data/model/HCMP_A/checkpoint_hcmp_A.h5'
python3 src/inference.py --test_path='/data/test/HCMP' --output_path='/data/output/HCMP' --test_filename='N_' --model_name='/data/model/HCMP_N/checkpoint_hcmp_N.h5'
