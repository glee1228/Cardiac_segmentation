mkdir /data/model/CHD_dia
mkdir /data/model/CHD_sys
mkdir /data/model/HCMP_A
mkdir /data/model/HCMP_N
python3 src/train.py --data_dir='/data/train/CHD' --checkpoint_dir='/data/model/CHD_dia' --image_filename='_dia.mha' --label_filename='_dia_M.mha'
python3 src/train.py --data_dir='/data/train/CHD' --checkpoint_dir='/data/model/CHD_sys' --image_filename='_sys.mha' --label_filename='_sys_M.mha'
python3 src/train.py --data_dir='/data/train/HCMP' --checkpoint_dir='/data/model/HCMP_A' --image_filename='A_' --label_filename='A_'
python3 src/train.py --data_dir='/data/train/HCMP' --checkpoint_dir='/data/model/HCMP_N' --image_filename='N_' --label_filename='N_'
