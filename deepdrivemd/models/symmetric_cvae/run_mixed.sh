while [ ! -f "$stop_file" ]; do
	files_seen=`wc -l < $global_path`
    	files_exist=`ls -1 $sim_data_dir | wc -l`
	echo files seen before - $files_seen
	echo files exist before- $files_exist
	if [[ $files_seen -eq $files_exist ]]
	then
		echo waiting! 
		sleep 60
	else
		echo running woohoo!!
		bash /data/shared/vishal/ANL-shared/cvae_gb/single_script python /data/shared/vishal/ANL-shared/cvae_gb/prepare_dataset.py
		bash /data/shared/vishal/ANL-shared/cvae_gb/run_script python /data/shared/vishal/ANL-shared/cvae_gb/run.py --mode train --cs-ip 10.80.0.100
		source /data/shared/msalim/env/bin/activate
		python /data/shared/vishal/ANL-shared/cvae_gb/transfer_global.py
		deactivate
		bash /data/shared/vishal/ANL-shared/cvae_gb/single_script python /data/shared/vishal/ANL-shared/cvae_gb/run.py --mode eval
	fi
done



