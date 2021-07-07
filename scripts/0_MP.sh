# Run experiments on MP dataset
# Author: Ryo Yonetani
# Affiliation: OMRON SINIC X
# 
# usage:
# $ sh 0_MP.sh | sh (if you want to run experiments sequentially)
# $ sh 0_MP.sh | parallel -j N --ungroup  (or run N experiments in parallel)

logdir=icml2021_mp
for dataset in `ls data/mpd/*_032_moore_c8.npz`
do
# A*
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m VanillaAstar -gb \"train.num_epochs = 1\" \"VanillaAstar.g_ratio = 0.5\" \"VanillaAstarPlanner.output_exp_instead_of_rel_exp = True\" 
# Weighted A*
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m VanillaAstar -gb \"train.num_epochs = 1\" \"VanillaAstar.g_ratio = 0.2\" 
# Best-first 
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m VanillaAstar -gb \"train.num_epochs = 1\" \"VanillaAstar.g_ratio = 0.0\" 
# SAIL
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m SAIL -gb \"SAIL.beta0 = 0.0\" 
# SAIL-SL
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m SAIL -gb \"SAIL.beta0 = 1.0\" 
# BB-A*
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m BBAstar -gb \"NeuralAstar.g_ratio = 0.5\" \"BBAstar.bbastar_lambda = 20.0\" \"BBAstarPlanner.dilate_gt = False\"
# Neural BF
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m NeuralAstar -gb \"NeuralAstar.g_ratio = 0.0\" \"NeuralAstarPlanner.dilate_gt = False\" \"NeuralAstar.detach_g = True\"
# Neural A*
echo python train.py -gf config/*.gin -d ${dataset} -w ${logdir} -m NeuralAstar -gb \"NeuralAstar.g_ratio = 0.5\" \"NeuralAstarPlanner.dilate_gt = False\" \"NeuralAstar.detach_g = True\"
done