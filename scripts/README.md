# Instruction on how to run the scripts
-----------------
* using tmux is preferred it already installed in your env it makea everything smooth (optional)
* run 'tmux new -t main' this will open a terminal within your dev cloud terminal (optional)
* If not using tmux just simply open your dev cloud terminal 
* run `pip install -r requirnments.txt`
* Let's start training run `python train.py --model_name <any seq2seq type model e.g. Salesforce/codet5-base> --exp_name <any experiment name e.g. experiment_1>`
* If you using tmux press ctrl+b and then d this will minimise your terminal
* To open again run `tmux a -t main` this resume your terminal without stopping training
* Navigate your final model and we can start generating result/submission
* run `python results.py --model_checkpoint <path_of_trained_model> --output <output_file_name>`
* This will infer the model and save your submission file and raw file for postprocessing in current directory
* Now run `python analysis.py --model_checkpoint <path_of_trained_mdoel>` to see what are the rows that gave wrong result and actual expected result/

## Further analysis of these results are available in notebook files (navigate there) and also covered during presentation (and in ppt file)